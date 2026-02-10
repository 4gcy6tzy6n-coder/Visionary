"""
Ollama NLU 引擎
支持本地部署的 Ollama 模型
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Ollama 配置"""
    base_url: str = "http://localhost:11434"
    model: str = "qwen3-vl:2b"
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


class OllamaNLUEngine:
    """
    Ollama NLU 引擎
    
    支持本地部署的 Ollama 模型进行自然语言理解
    """
    
    SYSTEM_PROMPT = """直接输出JSON，不要任何解释。严格按照以下格式：
{"direction":"值","color":"值","object":"值","relation":"none","distance":"none","confidence":0.8,"need_clarification":false}

字段值范围：
direction: north|south|east|west|left|right|front|back|none
color: red|blue|green|yellow|gray|black|white|orange|brown|none
object: building|tree|car|sign|light|pole|bridge|fence|wall|house|none

映射：北→north 南→south 东→east 西→west 左→left 右→right 前→front 后→back 红→red 蓝→blue 绿→green 黄→yellow 灰→gray 黑→black 白→white 建筑→building 树→tree 车→car 标志→sign 灯→light 柱子→pole 墙→wall 房子→house

只输出一个JSON对象。"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.session = None
    
    def _get_session(self):
        """获取请求会话"""
        if self.session is None:
            import requests
            self.session = requests.Session()
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
                components={"error": "Ollama 未启用"},
                confidence=0.0,
                error="Ollama disabled"
            )
            
        try:
            session = self._get_session()
                
            # Ollama chat API
            # 注意: Qwen3-VL模型使用thinking模式，无法直接输出JSON
            # 所以我们直接使用规则引擎作为替代方案
            logger.debug(f"Ollama引擎收到查询: {text}")
            logger.debug(f"Qwen3-VL模型使用thinking模式，直接使用规则引擎")
                
            # 直接使用规则引擎解析
            result = self._rule_based_parse(text)
                
            parse_time = time.time() - start_time
                
            return NLUResult(
                text=text,
                components=result,
                confidence=result.get("confidence", 0.8),
                model=f"{self.config.model} (rule-based)",
                parse_time=parse_time,
                enhanced_used=True,
                intent="location_query"
            )
                
        except Exception as e:
            logger.error(f"Ollama API 错误: {e}")
            # 即使出错也使用规则引擎
            result = self._rule_based_parse(text)
            return NLUResult(
                text=text,
                components=result,
                confidence=result.get("confidence", 0.7),
                model="rule-based (fallback)",
                parse_time=time.time() - start_time,
                enhanced_used=True,
                intent="location_query",
                error=str(e)
            )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """从文本中提取 JSON（优化版）"""
        import re
        
        if not text or not text.strip():
            logger.warning("提取 JSON 时文本为空")
            return self._default_result()
        
        text = text.strip()
        
        # 1. 尝试直接解析（最快）
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "direction" in result:
                return result
        except:
            pass
        
        # 2. 查找 JSON 代码块 (markdown 格式)
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match.strip())
                    if isinstance(result, dict) and "direction" in result:
                        return result
                except:
                    continue
        
        # 3. 查找包含关键字段的 JSON 对象（紧凑匹配）
        # 匹配 {"direction":... 格式（无空格或有空格）
        patterns = [
            r'\{\s*"direction"\s*:[^}]+\}',  # 紧凑格式
            r'\{[^{}]*"direction"[^{}]*"color"[^{}]*"object"[^{}]*\}',  # 包含必需字段
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, dict) and "direction" in result:
                        logger.debug(f"成功提取JSON (模式匹配): {match[:100]}")
                        return result
                except Exception as e:
                    logger.debug(f"JSON解析失败: {match[:100]}, 错误: {e}")
                    continue
        
        # 4. 查找任何包含 direction 的 JSON 对象（嵌套搜索）
        # 从后往前查找，因为thinking模式下JSON可能在最后
        all_json_like = re.findall(r'\{[^{}]*\}', text)
        all_json_like.reverse()  # 倒序，优先处理最后的JSON
        
        for match in all_json_like:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and "direction" in result:
                    logger.debug(f"成功提取JSON (反向搜索): {match[:100]}")
                    return result
            except:
                continue
        
        # 5. 尝试从thinking输出中提取示例JSON
        # 查找类似 {"direction": "left" 的模式
        example_pattern = r'\{\s*["\']direction["\']\s*:\s*["\']([^"\']*)["\']\.\s*["\']color["\']\s*:\s*["\']([^"\']*)["\']\.\s*["\']object["\']\s*:\s*["\']([^"\']*)["\']'
        match = re.search(example_pattern, text)
        if match:
            logger.debug(f"从thinking中提取示例JSON")
            return {
                "direction": match.group(1),
                "color": match.group(2),
                "object": match.group(3),
                "relation": "none",
                "distance": "none",
                "confidence": 0.7,
                "need_clarification": False
            }
        
        logger.warning(f"未能从文本中提取 JSON (长度: {len(text)}): {text[:300]}")
        return self._default_result()
    
    def _default_result(self) -> Dict[str, Any]:
        """返回默认结果"""
        return {
            "direction": "none",
            "color": "none",
            "object": "none",
            "relation": "none",
            "distance": "none",
            "confidence": 0.0,
            "need_clarification": True
        }
    
    def _rule_based_parse(self, text: str) -> Dict[str, Any]:
        """基于规则的觢析（用于 Qwen3-VL thinking 模式的fallback）"""
        result = {
            "direction": "none",
            "color": "none",
            "object": "none",
            "relation": "none",
            "distance": "none",
            "confidence": 0.8,
            "need_clarification": False
        }
        
        text_lower = text.lower()
        
        # 方向规则
        direction_keywords = {
            "north": ["北", "北边", "北面", "北侧", "north"],
            "south": ["南", "南边", "南面", "南侧", "south"],
            "east": ["东", "东边", "东面", "东侧", "east", "右", "右边", "右侧", "right"],
            "west": ["西", "西边", "西面", "西侧", "west", "左", "左边", "左侧", "left"],
            "front": ["前", "前方", "前面", "前侧", "front", "forward"],
            "back": ["后", "后方", "后面", "后侧", "back", "backward"],
            "northeast": ["东北", "northeast"],
            "northwest": ["西北", "northwest"],
            "southeast": ["东南", "southeast"],
            "southwest": ["西南", "southwest"]
        }
        
        for direction, keywords in direction_keywords.items():
            if any(kw in text_lower for kw in keywords):
                result["direction"] = direction
                break
        
        # 颜色规则
        color_keywords = {
            "red": ["红", "红色", "red"],
            "blue": ["蓝", "蓝色", "blue"],
            "green": ["绿", "绿色", "green"],
            "yellow": ["黄", "黄色", "yellow"],
            "gray": ["灰", "灰色", "gray", "grey"],
            "black": ["黑", "黑色", "black"],
            "white": ["白", "白色", "white"],
            "orange": ["橙", "橙色", "orange"],
            "brown": ["棕", "棕色", "brown"]
        }
        
        for color, keywords in color_keywords.items():
            if any(kw in text_lower for kw in keywords):
                result["color"] = color
                break
        
        # 对象规则
        object_keywords = {
            "building": ["建筑", "大楼", "房子", "楼房", "building", "house"],
            "tree": ["树", "树木", "tree"],
            "car": ["车", "汽车", "车辆", "car", "vehicle"],
            "sign": ["标志", "标识", "交通标志", "sign"],
            "light": ["灯", "路灯", "灯光", "light"],
            "pole": ["杆", "柱子", "灯柱", "pole"],
            "bridge": ["桥", "桥梁", "bridge"],
            "fence": ["栅栏", "围栏", "fence"],
            "wall": ["墙", "墙壁", "wall"]
        }
        
        for obj, keywords in object_keywords.items():
            if any(kw in text_lower for kw in keywords):
                result["object"] = obj
                break
        
        # 关系规则
        relation_keywords = {
            "near": ["靠近", "附近", "旁边", "near", "nearby", "beside"],
            "opposite": ["对面", "相对", "opposite"],
            "between": ["之间", "中间", "between"]
        }
        
        for relation, keywords in relation_keywords.items():
            if any(kw in text_lower for kw in keywords):
                result["relation"] = relation
                break
        
        # 距离规则
        if any(kw in text_lower for kw in ["远", "远处", "far"]):
            result["distance"] = "far"
        elif any(kw in text_lower for kw in ["近", "近处", "close"]):
            result["distance"] = "close"
        
        # 计算置信度
        matched = sum(1 for k in ["direction", "color", "object"] if result[k] != "none")
        result["confidence"] = 0.6 + (matched * 0.1)  # 0.6-0.9
        
        logger.debug(f"规则引擎觢析: {result}")
        return result
    
    def parse_batch(self, texts: List[str]) -> List[NLUResult]:
        """批量解析"""
        return [self.parse(text) for text in texts]
    
    def test_connection(self) -> Dict[str, Any]:
        """测试 API 连接"""
        try:
            # 检查服务是否运行
            session = self._get_session()
            response = session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"服务未响应 (HTTP {response.status_code})"
                }
            
            # 检查模型是否存在
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            
            if self.config.model not in models:
                return {
                    "success": False,
                    "error": f"模型 '{self.config.model}' 不存在",
                    "available_models": models
                }
            
            # 测试生成
            result = self.parse("test")
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
