"""
DeepSeek 大模型集成模块
用于 Text2Loc 位置描述解析
"""
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """模型提供商"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class DeepSeekConfig:
    """DeepSeek API 配置"""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 512
    temperature: float = 0.1
    timeout: int = 30
    enabled: bool = True


@dataclass
class NLUResult:
    """NLU 解析结果"""
    text: str
    components: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    model: str = "deepseek"
    parse_time: float = 0.0
    enhanced_used: bool = True
    intent: str = "location_query"
    error: Optional[str] = None
    need_clarification: bool = False
    clarification_question: Optional[str] = None


class DeepSeekClient:
    """
    DeepSeek API 客户端
    
    支持:
    - 位置描述解析
    - 方向识别
    - 颜色识别
    - 对象识别
    - 关系理解
    """
    
    SYSTEM_PROMPT = """你是一个专业的位置描述解析助手。你的任务是将用户的中英文位置描述转换为结构化的标准格式，并在信息不足时智能生成澄清问题。

## 输出格式
请严格按照以下 JSON 格式输出：
{
    "direction": "方向词(东/南/西/北/东北/西北/东南/西南/无)",
    "color": "颜色词(红/蓝/绿/黄/灰/黑/白/紫/橙/棕/无)",
    "object": "目标对象(大楼/建筑/树/山/路/灯/车/桥/河/无)",
    "relation": "关系词(靠近/旁边/对面/附近/在...北边/在...东边/无)",
    "distance": "距离词(近/远/附近/无)",
    "confidence": 0.85,
    "need_clarification": false,
    "clarification_question": ""
}

## 规则
1. 方向词: north/east/south/west/northeast/northwest/southeast/southwest 或 北/东/南/西/东北/西北/东南/西南
2. 颜色词: red/blue/green/yellow/gray/black/white/purple/orange/brown 或 红/蓝/绿/黄/灰/黑/白/紫/橙/棕
3. 对象词: building/tree/mountain/road/lamp/car/bridge/river 或 大楼/树/山/路/灯/车/桥/河
4. 如果没有识别到某项，设为 "none"
5. confidence 范围 0-1，根据识别确定性设置
6. 如果信息不完整需要追问，设置 need_clarification=true，并生成个性化的 clarification_question

## 智能推理规则（重要）
你必须根据用户的描述智能推理方向和位置信息，而不是直接询问。要善于从各种线索中提取隐含的方向和位置信息。

### 时间和太阳推理
- 上午（6-12点）+ 面朝太阳 = 面朝东方（east）
- 上午（6-12点）+ 背对太阳 = 面朝西方（west）
- 下午（12-18点）+ 面朝太阳 = 面朝西方（west）
- 下午（12-18点）+ 背对太阳 = 面朝东方（east）
- 中午（11-13点）太阳在南方（北半球），背对太阳 = 面朝北方
- 日出时分 = 面朝东方，日落时分 = 面朝西方
- 上午影子指向西方，下午影子指向东方

### 环境特征推理
- "风吹过来"、"风从...来"：结合季节和地理常识推理方向
- "晒不到太阳"、"阴凉"：可能在建筑物的北侧或阴影区域
- "很晒"、"阳光直射"：可能在开阔地带或南侧
- "能看到日落"：面朝西方；"能看到日出"：面朝东方
- "背靠大山"、"山在后面"：面朝与山相反的方向
- "面向大海/河流"：面朝水域方向
- "在树荫下"：靠近树木，可能在树的某个方向

### 声音和气味推理
- "听到...声音从...传来"：根据声源方向定位
- "闻到...味道"：根据气味来源方向定位
- "车声从左边传来"、"右边有流水声"：利用声音方向

### 建筑物和地标推理
- "在...和...之间"：位于两个物体的中间位置
- "在...对面"、"隔着马路"：相对位置关系
- "在...旁边"、"紧挨着"：邻近关系
- "在...拐角"、"路口"：交叉位置
- "在...前面/后面"：需要明确参考系的朝向
- "二楼"、"顶层"：垂直方向信息

### 相对方向推理
- "A在B的左边/右边"：需要根据B的朝向判断
- "A在B的前方/后方"：以B的朝向为参考
- "顺时针方向"、"逆时针方向"：旋转方向
- "对角线位置"：对角关系

### 运动和路径推理
- "从...走过来"、"往...走"：运动方向和当前位置
- "刚转弯"、"拐角处"：路径变化点
- "上坡"、"下坡"：地形变化
- "沿着...走"：线性参考

### 人群和活动推理
- "人很多"、"热闹"：可能在商业区、广场、入口
- "很安静"：可能在住宅区、公园深处、小巷
- "看到排队"：可能在热门地点、入口、售票处
- "有施工"、"在维修"：临时地标

### 天气和季节推理
- "雪还没化"：背阴处或北侧
- "落叶很多"：树木下方或南侧（落叶树）
- "积水"：低洼处或排水不畅区域
- "有冰"：背阴处或北侧

## 澄清问题生成规则
当 need_clarification=true 时，根据已识别的信息智能生成问题：
- 如果缺少方向："您提到在{object}附近，请问具体在哪个方向呢？"
- 如果缺少对象："您在{direction}边，附近有什么标志性建筑或物体吗？"
- 如果都缺少："为了更准确地定位，您能描述一下周围的环境吗？比如有什么建筑物、在什么方向？"
- 如果信息模糊："您描述的位置有点宽泛，能再具体一些吗？比如距离{object}有多远？"

## 示例
- "我在红色大楼的北边" -> {"direction":"north","color":"red","object":"building","confidence":0.9,"need_clarification":false}
- "我现在面朝太阳，现在是上午" -> {"direction":"east","confidence":0.85,"need_clarification":true,"clarification_question":"您面朝东方（上午太阳在东方），请问附近有什么标志性建筑或物体吗？"}
- "我在树荫下，能看到日落" -> {"direction":"west","object":"tree","confidence":0.75,"need_clarification":true,"clarification_question":"您在树下且面朝西方（能看到日落），请问这棵树在什么物体附近？"}
- "我听到车声从左边传来，前面有座桥" -> {"direction":"east","object":"bridge","confidence":0.7,"need_clarification":true,"clarification_question":"您面朝东方，前方有桥，请问附近还有其他标志性建筑吗？"}

请直接输出 JSON，不要有其他内容。"""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.session = None
    
    def _get_session(self):
        """获取请求会话"""
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json; charset=utf-8"
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
                components={"error": "DeepSeek 未启用"},
                confidence=0.0,
                error="DeepSeek disabled"
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
                "temperature": self.config.temperature,
                "response_format": {"type": "json_object"}
            }
            
            response = session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            
            # 解析 JSON
            result = json.loads(content)
            
            parse_time = time.time() - start_time
            
            # 提取澄清问题
            need_clarification = result.get("need_clarification", False)
            clarification_question = result.get("clarification_question", "")
            
            # 如果没有生成澄清问题但 need_clarification 为 true，使用默认问题
            if need_clarification and not clarification_question:
                direction = result.get("direction", "none")
                obj = result.get("object", "none")
                if direction != "none" and obj == "none":
                    clarification_question = f"您在{direction}边，附近有什么标志性建筑或物体吗？"
                elif direction == "none" and obj != "none":
                    clarification_question = f"您提到在{obj}附近，请问具体在哪个方向呢？"
                else:
                    clarification_question = "为了更准确地定位，您能描述一下周围的环境吗？比如有什么建筑物、在什么方向？"
            
            return NLUResult(
                text=text,
                components={
                    "direction": result.get("direction", "none"),
                    "color": result.get("color", "none"),
                    "object": result.get("object", "none"),
                    "relation": result.get("relation", "none"),
                    "distance": result.get("distance", "none"),
                    "confidence": result.get("confidence", 0.8),
                    "need_clarification": need_clarification,
                    "clarification_question": clarification_question
                },
                confidence=result.get("confidence", 0.8),
                model=self.config.model,
                parse_time=parse_time,
                enhanced_used=True,
                intent="location_query",
                need_clarification=need_clarification,
                clarification_question=clarification_question
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"DeepSeek JSON 解析错误: {e}")
            return NLUResult(
                text=text,
                components={"error": f"JSON解析失败: {str(e)}"},
                confidence=0.0,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"DeepSeek API 错误: {e}")
            return NLUResult(
                text=text,
                components={"error": str(e)},
                confidence=0.0,
                error=str(e)
            )
    
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


class DeepSeekNLUEngine:
    """
    DeepSeek NLU 引擎
    集成到 Text2Loc 系统中
    """
    
    def __init__(self, config: Optional[DeepSeekConfig] = None):
        """
        初始化引擎
        
        Args:
            config: DeepSeek 配置，如果为 None 则使用默认配置
        """
        if config is None:
            config = DeepSeekConfig()
        self.config = config
        self.client = DeepSeekClient(config)
        logger.info(f"DeepSeek NLU 引擎初始化: model={config.model}")
    
    def parse(self, text: str) -> NLUResult:
        """
        解析自然语言
        
        Args:
            text: 用户输入
            
        Returns:
            NLUResult: 解析结果
        """
        return self.client.parse(text)
    
    def set_api_key(self, api_key: str):
        """设置 API Key"""
        self.config.api_key = api_key
        self.client = DeepSeekClient(self.config)
    
    def enable(self, enabled: bool = True):
        """启用/禁用"""
        self.config.enabled = enabled


def create_deepseek_engine(api_key: str = "", model: str = "deepseek-chat") -> DeepSeekNLUEngine:
    """
    创建 DeepSeek 引擎的便捷函数
    
    Args:
        api_key: DeepSeek API Key
        model: 模型名称
        
    Returns:
        DeepSeekNLUEngine 实例
    """
    config = DeepSeekConfig(
        api_key=api_key,
        model=model,
        enabled=True
    )
    return DeepSeekNLUEngine(config)


if __name__ == "__main__":
    # 快速测试
    print("=" * 60)
    print("🧪 DeepSeek NLU 引擎测试")
    print("=" * 60)
    
    # 配置（用户需要填入自己的 API Key）
    config = DeepSeekConfig(
        api_key="YOUR_API_KEY_HERE",  # 替换为实际 API Key
        model="deepseek-chat",
        timeout=30
    )
    
    engine = DeepSeekNLUEngine(config)
    
    test_queries = [
        "我在红色大楼的北边",
        "I am north of a red building",
        "树林靠近山的位置",
        "交通灯的东边有一个停车区域"
    ]
    
    for query in test_queries:
        print(f"\n📝 查询: {query}")
        result = engine.parse(query)
        
        if result.error:
            print(f"   ❌ 错误: {result.error}")
        else:
            print(f"   ✅ 成功")
            print(f"      方向: {result.components.get('direction')}")
            print(f"      颜色: {result.components.get('color')}")
            print(f"      对象: {result.components.get('object')}")
            print(f"      置信度: {result.confidence:.2f}")
            print(f"      耗时: {result.parse_time:.3f}秒")
