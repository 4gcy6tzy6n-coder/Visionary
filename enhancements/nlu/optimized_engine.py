"""
Text2Loc NLU 引擎 - 智能解析版
专门将自然语言转换为Text2Loc标准格式
通过Qwen模型分析理解，转换为标准格式后传递给原始Text2Loc系统
"""

import json
import logging
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NLUResult:
    text: str
    components: Dict[str, Any]
    confidence: float
    model: str
    parse_time: float
    enhanced_used: bool
    need_clarification: bool = False  # 是否需要澄清
    clarification_question: Optional[str] = None  # 澄清问题
    intent: Optional[str] = None  # 意图类型
    error: Optional[str] = None  # 错误信息


@dataclass
class StandardFormat:
    """Text2Loc标准格式"""
    object_label: str  # 对象类别（如：building, tree, parking）
    object_color: str  # 颜色（如：red, blue, green）
    direction: str  # 方向（如：north, south, east, west, on-top）
    offset: Optional[List[float]] = None  # 偏移量 [x, y]
    cell_id: Optional[str] = None  # 单元格ID
    pose_id: Optional[str] = None  # 姿态ID
    description: Optional[str] = None  # 描述文本
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_label": self.object_label,
            "object_color": self.object_color,
            "direction": self.direction,
            "offset": self.offset,
            "cell_id": self.cell_id,
            "pose_id": self.pose_id,
            "description": self.description,
        }


@dataclass
class NLUConfig:
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen3-vl:2b"
    timeout: int = 600  # 10分钟超时（模型训练中需要更长时间）
    mock_mode: bool = False
    confidence_threshold: float = 0.6
    enable_dialog: bool = True
    max_history: int = 10


class OptimizedNLUEngine:
    """
    智能NLU引擎 - Text2Loc标准格式转换器
    通过Qwen模型分析理解自然语言，转换为Text2Loc标准格式
    支持传递给原始Text2Loc系统进行真正的定位
    """

    # 意图类型
    INTENT_TYPES = {
        "query_location": ["在哪里", "位置", "地方", "方位", "方向", "locate", "find"],
        "describe_location": ["我在", "位于", "处于", "站在", "坐在", "在"],
        "ask_direction": ["怎么走", "方向", "朝向", "面向", "direction"],
        "find_nearby": ["附近", "旁边", "靠近", "邻近", "nearby", "close"],
        "confirm": ["是", "对", "好的", "没错", "确认", "yes", "ok"],
        "clarify": ["哪个", "什么", "哪里", "how", "what", "which"],
        "greeting": ["你好", "hello", "hi", "您好", "早上好", "晚上好"],
    }

    # Text2Loc标准对象类别（22种）
    STANDARD_OBJECTS = {
        "box": ["箱子", "盒子", "方块", "box"],
        "bridge": ["桥", "桥梁", "天桥", "bridge"],
        "building": ["大楼", "建筑", "建筑物", "楼房", "大厦", "房子", "房屋", "building"],
        "fence": ["围墙", "栅栏", "栏杆", "fence"],
        "garage": ["车库", "停车库", "garage"],
        "guard rail": ["护栏", "防护栏", "安全栏", "guard rail"],
        "lamp": ["路灯", "灯具", "lamp"],
        "pad": ["垫子", "平台", "基座", "pad"],
        "parking": ["停车场", "停车位", "车位", "泊车位", "parking"],
        "pole": ["柱子", "灯柱", "电线杆", "杆子", "立柱", "pole"],
        "road": ["道路", "马路", "公路", "road"],
        "sidewalk": ["人行道", "步道", "便道", "sidewalk"],
        "smallpole": ["小柱子", "短杆", "小杆", "smallpole"],
        "stop": ["停止", "停止标志", "停车", "stop"],
        "terrain": ["地形", "地面", "土地", "terrain"],
        "traffic light": ["红绿灯", "交通灯", "信号灯", "traffic light"],
        "traffic sign": ["交通标志", "路标", "指示牌", "标牌", "traffic sign"],
        "trash bin": ["垃圾桶", "垃圾箱", "废物箱", "trash bin"],
        "tunnel": ["隧道", "地道", "隧洞", "tunnel"],
        "vegetation": ["植被", "植物", "草木", "vegetation"],
        "vending machine": ["自动售货机", "贩卖机", "售货机", "vending machine"],
        "wall": ["墙", "墙壁", "墙体", "wall"],
    }

    # Text2Loc标准颜色（8种）
    STANDARD_COLORS = {
        "dark-green": ["深绿色", "墨绿", "dark-green"],
        "gray": ["灰色", "灰", "gray"],
        "gray-green": ["灰绿色", "gray-green"],
        "bright-gray": ["亮灰色", "浅灰", "bright-gray"],
        "gray": ["灰色", "灰", "gray"],  # 重复但保持兼容
        "black": ["黑色", "黑", "black"],
        "green": ["绿色", "绿", "green"],
        "beige": ["米色", "浅褐色", "beige"],
    }

    # Text2Loc标准方向（5种）
    STANDARD_DIRECTIONS = {
        "north": ["北", "北方", "北边", "北侧", "前方", "前边", "前面", "前侧", "north"],
        "south": ["南", "南方", "南边", "南侧", "后方", "后边", "后面", "后侧", "south"],
        "east": ["东", "东方", "东边", "东侧", "右侧", "右边", "right", "east"],
        "west": ["西", "西方", "西边", "西侧", "左侧", "左边", "left", "west"],
        "on-top": ["上", "上方", "上面", "顶部", "on-top", "above", "over"],
    }

    # 扩展的方向映射（更全面的中文方向词）
    DIRECTIONS = {
        "north": ["北", "北方", "北边", "北侧", "前方", "前边", "前面", "前侧", "北面", "forward", "front"],
        "south": ["南", "南方", "南边", "南侧", "后方", "后边", "后面", "后侧", "南面", "backward", "back"],
        "east": ["东", "东方", "东边", "东侧", "右侧", "右边", "右面", "right", "east"],
        "west": ["西", "西方", "西边", "西侧", "左侧", "左边", "左面", "left", "west"],
        "northeast": ["东北", "东北方", "东北边", "东北角", "东北侧", "北东", "northeast"],
        "southeast": ["东南", "东南方", "东南边", "东南角", "东南侧", "南东", "southeast"],
        "southwest": ["西南", "西南方", "西南边", "西南角", "西南侧", "南西", "southwest"],
        "northwest": ["西北", "西北方", "西北边", "西北角", "西北侧", "北西", "northwest"],
        "on_top": ["上", "上方", "上面", "顶部", "atop", "over", "above"],
        "below": ["下", "下方", "下面", "底部", "under", "beneath"],
    }

    # 扩展的颜色映射
    COLORS = {
        "red": ["红色", "红", "赤色", "红红", "深红"],
        "blue": ["蓝色", "蓝", "天蓝", "深蓝", "浅蓝"],
        "green": ["绿色", "绿", "青色", "深绿", "浅绿", "草绿色"],
        "yellow": ["黄色", "黄", "金黄色", "淡黄"],
        "gray": ["灰色", "灰", "深灰", "浅灰", "银灰色"],
        "black": ["黑色", "黑", "深黑", "漆黑"],
        "white": ["白色", "白", "洁白", "乳白"],
        "brown": ["棕色", "褐", "褐色", "咖啡色"],
        "orange": ["橙色", "橙", "橘色", "橘黄"],
        "purple": ["紫色", "紫", "深紫", "浅紫"],
    }

    # 扩展的对象映射（包含更多中文场景对象）
    OBJECTS = {
        # 原有类别
        "building": ["大楼", "建筑", "建筑物", "楼房", "大厦", "房子", "房屋", "building"],
        "pole": ["柱子", "灯柱", "电线杆", "杆子", "立柱", "pole"],
        "parking": ["停车场", "停车位", "车位", "泊车位", "parking"],
        "sign": ["标志", "交通标志", "路标", "指示牌", "标牌", "sign"],
        "light": ["灯", "路灯", "照明灯", "灯具", "light"],
        "car": ["汽车", "车辆", "小车", "轿车", "机动车", "car"],
        "tree": ["树", "树木", "大树", "树林", "forest", "woods", "tree", "林木"],
        "box": ["箱子", "盒子", "方块", "box"],
        "bridge": ["桥", "桥梁", "天桥", "bridge"],
        "fence": ["围墙", "栅栏", "栏杆", "fence"],
        "garage": ["车库", "停车库", "garage"],
        "guard rail": ["护栏", "防护栏", "安全栏", "guard rail"],
        "road": ["道路", "马路", "公路", "road"],
        "sidewalk": ["人行道", "步道", "便道", "sidewalk"],
        "terrain": ["地形", "地面", "土地", "terrain"],
        "traffic light": ["红绿灯", "交通灯", "信号灯", "traffic light"],
        "trash bin": ["垃圾桶", "垃圾箱", "废物箱", "trash bin"],
        "tunnel": ["隧道", "地道", "隧洞", "tunnel"],
        "wall": ["墙", "墙壁", "墙体", "wall"],
        
        # 新增自然场景对象
        "mountain": ["山", "山峰", "山丘", "丘陵", "土堆", "mountain", "hill", "山脚", "山坡"],
        "vegetation": ["植被", "植物", "草木", "草丛", "vegetation"],
        "grass": ["草地", "草坪", "草", "草原", "grass"],
        "water": ["水", "河流", "湖", "池塘", "水域", "water", "river", "lake"],
        "rock": ["石头", "岩石", "石块", "岩石", "rock", "stone"],
        "path": ["小路", "路径", "道路", "通道", "path", "trail", "小径"],
        "entrance": ["入口", "门口", "大门", "入口处", "entrance", "door"],
        "corner": ["角落", "拐角", "墙角", "corner"],
        "junction": ["路口", "交叉口", "交汇处", "junction", "intersection"],
    }

    # 扩展的空间关系映射
    RELATIONS = {
        "near": ["靠近", "邻近", "附近", "旁边", "近", "near", "beside", "close to", "next to"],
        "between": ["之间", "中间", "当中", "之间", "between"],
        "above": ["上方", "上面", "顶部", "正上方", "above", "over", "on top of"],
        "below": ["下方", "下面", "底部", "底下", "below", "under", "beneath"],
        "in_front_of": ["前面", "前方", "正前方", "前侧", "in front of", "before"],
        "behind": ["后面", "后方", "后侧", "背后", "behind", "after"],
        "left_of": ["左边", "左侧", "左方", "left of"],
        "right_of": ["右边", "右侧", "右方", "right of"],
        "at": ["在", "位于", "处于", "at", "located at"],
        "facing": ["面向", "朝向", "面对", "facing", "facing towards"],
    }

    # 距离词到数值的映射
    DISTANCE_WORDS = {
        "很近": 2,
        "非常近": 1,
        "近": 3,
        "不远": 5,
        "附近": 5,
        "约5米": 5,
        "约10米": 10,
        "约15米": 15,
        "约20米": 20,
        "几十米": 30,
        "百米": 100,
        "远处": 50,
    }

    def __init__(self, config: Optional[NLUConfig] = None):
        self.config = config or NLUConfig()
        self.session = requests.Session() if not self.config.mock_mode else None
        
        # 对话上下文管理
        self.dialog_contexts: Dict[str, DialogContext] = {}
        
        if not self.config.mock_mode:
            self._warmup()
        
        logger.info(f"优化版NLU引擎初始化: mock={self.config.mock_mode}, dialog={self.config.enable_dialog}")

    def _warmup(self):
        """预热模型"""
        try:
            logger.info(f"预热模型 {self.config.model_name}...")
            self.session.post(
                f"{self.config.ollama_url}/api/generate",
                json={"model": self.config.model_name, "prompt": "hello", "stream": False},
                timeout=120
            )
            logger.info("✅ 模型预热成功")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def parse(self, text: str) -> NLUResult:
        """
        解析自然语言查询（智能解析版）
        
        Args:
            text: 用户输入的自然语言文本
            
        Returns:
            NLU解析结果（包含Text2Loc标准格式）
        """
        start_time = time.time()
        
        try:
            # 检测意图
            intent = self._detect_intent(text)
            
            # 处理问候语
            if intent == "greeting":
                return self._handle_greeting(text, start_time)
            
            # 解析查询
            if self.config.mock_mode:
                result = self._mock_parse(text)
            else:
                result = self._api_parse(text)
            
            parse_time = time.time() - start_time
            
            return NLUResult(
                text=text,
                components=result["components"],
                confidence=result["confidence"],
                model=self.config.model_name,
                parse_time=parse_time,
                enhanced_used=True,
                intent=intent
            )
        except Exception as e:
            logger.error(f"解析失败: {e}")
            parse_time = time.time() - start_time
            
            return NLUResult(
                text=text,
                components={"error": str(e)},
                confidence=0.0,
                model=self.config.model_name,
                parse_time=parse_time,
                enhanced_used=False,
                error=str(e)
            )

    def _api_parse(self, text: str, context: Optional[Any] = None) -> Dict[str, Any]:
        """调用API解析"""
        prompt = self._create_prompt(text)
        
        response = self.session.post(
            f"{self.config.ollama_url}/api/generate",
            json={
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "max_tokens": 500}
            },
            timeout=self.config.timeout
        )
        
        if response.status_code == 200:
            response_text = response.json().get("response", "")
            json_text = self._extract_json(response_text)
            return self._parse_response(json_text)
        
        raise Exception(f"API错误: {response.status_code}")

    def _create_prompt(self, text: str) -> str:
        """创建优化的提示词 - 增强版"""
        
        return f"""你是一个专业的空间位置解析专家，擅长理解自然语言中的位置描述信息。

【任务】分析用户输入并提取精确的位置要素

【标准对象类别】(22种，请选择最接近的):
- 建筑类: building(建筑/大楼), bridge(桥梁), tunnel(隧道), wall(墙), garage(车库)
- 设施类: parking(停车场), sign(标志), light(灯), lamp(路灯), pole(柱子), smallpole(小柱子)
- 安全类: fence(围栏), guard rail(护栏), stop(停止标志)
- 道路类: road(道路), sidewalk(人行道), pad(平台)
- 设备类: traffic light(交通灯), traffic sign(交通标志), trash bin(垃圾桶), vending machine(售货机)
- 环境类: terrain(地形), vegetation(植被), box(箱子)

【标准颜色】(8种，请选择最接近的):
- dark-green(深绿), gray(灰色), gray-green(灰绿), bright-gray(亮灰), black(黑色), green(绿色), beige(米色)

【标准方向】(5种，请精确判断):
- north(北/前方/前侧) - 包含:北、前、前方、前面、北侧、北边
- south(南/后方/后侧) - 包含:南、后、后方、后面、南侧、南边  
- east(东/右侧) - 包含:东、右、右侧、右边、东侧、东边
- west(西/左侧) - 包含:西、左、左侧、左边、西侧、西边
- on-top(上方) - 包含:上、上方、上面、顶部、之上

【用户输入】: "{text}"

【分析步骤】:
1. 识别所有提到的对象(建筑、物体等)
2. 识别所有提到的颜色描述
3. 识别所有方向信息(北/南/东/西/上方等)
4. 计算整体理解置信度(0.0-1.0)

【输出要求】严格JSON格式，不要任何额外文字:
{{
    "object_label": "最匹配的标准对象类别",
    "object_color": "最匹配的标准颜色",
    "direction": "最匹配的标准方向",
    "confidence": 0.85,
    "explanation": "简要说明选择理由"
}}

注意:
- 如果不确定某个字段,选择最接近的标准值
- confidence应真实反映理解程度
- 优先考虑用户明确提到的信息
- 对于中文方向词,准确映射到标准英文方向"""

    def _extract_json(self, text: str) -> str:
        """从响应中提取JSON"""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text

    def _parse_response(self, json_text: str) -> Dict[str, Any]:
        """解析API响应（Text2Loc标准格式）"""
        try:
            data = json.loads(json_text)
            
            # 提取标准格式信息
            components = {
                "object_label": {"value": data.get("object_label", "building"), "confidence": data.get("confidence", 0.5)},
                "object_color": {"value": data.get("object_color", "gray"), "confidence": data.get("confidence", 0.5)},
                "direction": {"value": data.get("direction", "north"), "confidence": data.get("confidence", 0.5)},
            }
            
            confidence = data.get("confidence", 0.5)
            
            return {
                "components": components,
                "confidence": round(confidence, 3),
                "explanation": data.get("explanation", "")
            }
        except json.JSONDecodeError:
            return self._mock_parse(json_text)

    def _mock_parse(self, text: str) -> Dict[str, Any]:
        """
        增强型解析（用于测试和回退）
        使用改进的算法提升识别准确性
        """
        text_lower = text.lower()
        text_chinese = text
        
        # 1. 增强对象检测 - 支持多关键词和优先级
        object_label = "building"  # 默认
        object_conf = 0.3  # 降低默认置信度
        matched_length = 0  # 记录匹配长度，优先长关键词
        
        for std_obj, keywords in self.STANDARD_OBJECTS.items():
            for keyword in keywords:
                if keyword in text_chinese or keyword in text_lower:
                    # 优先选择更长的关键词匹配(更具体)
                    if len(keyword) > matched_length:
                        object_label = std_obj
                        object_conf = min(0.95, 0.75 + len(keyword) * 0.03)
                        matched_length = len(keyword)
        
        # 2. 增强颜色检测 - 使用更智能的匹配
        object_color = "gray"  # 默认
        color_conf = 0.3  # 降低默认置信度
        color_matched_length = 0
        
        # 扩展颜色映射到Text2Loc标准颜色
        color_mapping = {
            "红": "gray",  # Text2Loc没有红色，映射到gray
            "蓝": "gray",
            "黄": "beige",
            "白": "bright-gray",
            "深绿": "dark-green",
            "墨绿": "dark-green",
            "浅灰": "bright-gray",
            "亮灰": "bright-gray"
        }
        
        for std_color, keywords in self.STANDARD_COLORS.items():
            for keyword in keywords:
                if keyword in text_chinese or keyword in text_lower:
                    if len(keyword) > color_matched_length:
                        object_color = std_color
                        color_conf = min(0.90, 0.70 + len(keyword) * 0.04)
                        color_matched_length = len(keyword)
        
        # 尝试颜色映射
        for cn_color, std_color in color_mapping.items():
            if cn_color in text_chinese:
                object_color = std_color
                color_conf = 0.85
                break
        
        # 3. 增强方向检测 - 多层次匹配
        direction = "north"  # 默认
        direction_conf = 0.3  # 降低默认置信度  
        direction_matched_length = 0
        
        # 方向权重 - 更精确的匹配
        direction_priority = {
            "north": ["北侧", "北边", "北方", "前侧", "前方", "前面", "北", "前"],
            "south": ["南侧", "南边", "南方", "后侧", "后方", "后面", "南", "后"],
            "east": ["东侧", "东边", "东方", "右侧", "右边", "东", "右"],
            "west": ["西侧", "西边", "西方", "左侧", "左边", "西", "左"],
            "on-top": ["上方", "顶部", "之上", "上面", "上"]
        }
        
        for std_dir, keywords in direction_priority.items():
            for keyword in keywords:
                if keyword in text_chinese:
                    if len(keyword) > direction_matched_length:
                        direction = std_dir
                        # 根据关键词长度和位置调整置信度
                        direction_conf = min(0.95, 0.80 + len(keyword) * 0.03)
                        direction_matched_length = len(keyword)
        
        # 如果没有匹配，尝试英文
        if direction_matched_length == 0:
            for std_dir, keywords in self.STANDARD_DIRECTIONS.items():
                for keyword in keywords:
                    if keyword in text_lower and len(keyword) >= 3:
                        direction = std_dir
                        direction_conf = 0.85
                        break
        
        # 4. 智能置信度计算
        # 根据实际匹配情况动态计算
        matched_fields = sum([1 for conf in [object_conf, color_conf, direction_conf] if conf > 0.5])
        
        if matched_fields >= 3:
            # 三个字段都匹配，高置信度
            confidence = (object_conf * 0.4 + color_conf * 0.3 + direction_conf * 0.3)
        elif matched_fields == 2:
            # 两个字段匹配，中等置信度
            confidence = (object_conf + color_conf + direction_conf) / 3 * 0.9
        else:
            # 只有一个字段匹配，低置信度
            confidence = (object_conf + color_conf + direction_conf) / 3 * 0.7
        
        # 根据查询长度调整置信度
        if len(text_chinese) < 5:
            confidence *= 0.8  # 查询太短降低置信度
        elif len(text_chinese) > 20:
            confidence *= 1.05  # 查询详细提升置信度
        
        confidence = min(confidence, 0.95)  # 限制最大值
        
        # 5. 生成详细解释
        explanation_parts = []
        if object_conf > 0.5:
            explanation_parts.append(f"对象:{object_label}(置信度{object_conf:.2f})")
        if color_conf > 0.5:
            explanation_parts.append(f"颜色:{object_color}(置信度{color_conf:.2f})")
        if direction_conf > 0.5:
            explanation_parts.append(f"方向:{direction}(置信度{direction_conf:.2f})")
        
        explanation = "; ".join(explanation_parts) if explanation_parts else "使用默认值"
        
        return {
            "components": {
                "object_label": {"value": object_label, "confidence": round(object_conf, 3)},
                "object_color": {"value": object_color, "confidence": round(color_conf, 3)},
                "direction": {"value": direction, "confidence": round(direction_conf, 3)},
            },
            "confidence": round(confidence, 3),
            "explanation": explanation
        }
    
    def _detect_all_objects(self, text: str) -> List[Dict]:
        """检测所有对象（返回列表）"""
        found = []
        text_lower = text.lower()
        
        for obj, keywords in self.OBJECTS.items():
            for keyword in keywords:
                if keyword in text_lower or keyword in text:
                    found.append({
                        "value": obj,
                        "confidence": 0.85
                    })
                    break
        
        return found

    def _detect_direction(self, text: str) -> Optional[Dict]:
        """检测方向"""
        for direction, keywords in self.DIRECTIONS.items():
            for keyword in keywords:
                if keyword in text:
                    return {
                        "value": direction,
                        "confidence": 0.9 if direction not in ["northeast", "southeast", "southwest", "northwest"] else 0.85
                    }
        return None

    def _detect_color(self, text: str) -> Optional[Dict]:
        """检测颜色"""
        for color, keywords in self.COLORS.items():
            for keyword in keywords:
                if keyword in text:
                    return {
                        "value": color,
                        "confidence": 0.85
                    }
        return None

    def _detect_object(self, text: str) -> Optional[Dict]:
        """检测对象（支持多对象检测，返回最相关的对象）"""
        found_objects = []
        
        for obj, keywords in self.OBJECTS.items():
            for keyword in keywords:
                if keyword in text:
                    found_objects.append((obj, len(keyword)))  # 使用关键词长度作为相关性权重
        
        if found_objects:
            # 按相关性排序（关键词越长越具体）
            found_objects.sort(key=lambda x: x[1], reverse=True)
            # 返回最相关的对象
            obj_name, _ = found_objects[0]
            return {
                "value": obj_name,
                "confidence": 0.85
            }
        
        return None

    def _detect_relation(self, text: str) -> Optional[Dict]:
        """检测空间关系"""
        for relation, keywords in self.RELATIONS.items():
            for keyword in keywords:
                if keyword in text:
                    return {
                        "value": relation,
                        "confidence": 0.8
                    }
        return None

    def _detect_distance(self, text: str) -> Optional[Dict]:
        """检测距离"""
        # 数字+米模式
        match = re.search(r'(\d+(?:\.\d+)?)\s*米', text)
        if match:
            return {
                "value": float(match.group(1)),
                "confidence": 0.9
            }
        
        # 距离词模式
        for word, value in self.DISTANCE_WORDS.items():
            if word in text:
                return {
                    "value": value,
                    "confidence": 0.6
                }
        
        return None
    
    def _detect_intent(self, text: str) -> Optional[str]:
        """检测用户意图"""
        text_lower = text.lower()
        
        for intent, keywords in self.INTENT_TYPES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent
        
        # 默认意图
        if any(word in text for word in ["我在", "位于", "处于", "站在"]):
            return "describe_location"
        
        return "query_location"
    
    def _handle_greeting(self, text: str, start_time: float) -> NLUResult:
        """处理问候语"""
        parse_time = time.time() - start_time
        
        greetings = [
            "你好！我是Text2Loc智能助手，可以帮你定位位置。请告诉我你的位置描述，例如：'我在红色大楼的北侧'",
            "您好！欢迎使用Text2Loc。请描述您的位置，我会帮您解析。",
            "你好！可以告诉我你现在在哪里吗？比如'我在一棵大树旁边'。"
        ]
        
        import random
        greeting_response = random.choice(greetings)
        
        return NLUResult(
            text=text,
            components={"greeting": greeting_response},
            confidence=1.0,
            model=self.config.model_name,
            parse_time=parse_time,
            enhanced_used=True,
            intent="greeting"
        )
    
    def _map_to_standard_format(self, components: Dict[str, Any]) -> StandardFormat:
        """
        将解析结果映射到Text2Loc标准格式
        
        Args:
            components: 解析出的组件
            
        Returns:
            Text2Loc标准格式
        """
        # 映射对象
        object_label = "building"  # 默认值
        if components.get("object"):
            obj_value = components["object"].get("value")
            if isinstance(obj_value, list):
                # 取第一个对象
                obj_value = obj_value[0] if obj_value else None
            
            if isinstance(obj_value, dict):
                obj_value = obj_value.get("value")
            
            # 查找最匹配的标准对象
            if obj_value:
                for std_obj, keywords in self.STANDARD_OBJECTS.items():
                    if obj_value in keywords or obj_value == std_obj:
                        object_label = std_obj
                        break
        
        # 映射颜色
        object_color = "gray"  # 默认值
        if components.get("color"):
            color_value = components["color"].get("value")
            if color_value:
                for std_color, keywords in self.STANDARD_COLORS.items():
                    if color_value in keywords or color_value == std_color:
                        object_color = std_color
                        break
        
        # 映射方向
        direction = "north"  # 默认值
        if components.get("direction"):
            dir_value = components["direction"].get("value")
            if dir_value:
                for std_dir, keywords in self.STANDARD_DIRECTIONS.items():
                    if dir_value in keywords or dir_value == std_dir:
                        direction = std_dir
                        break
        
        # 生成描述
        description = f"{object_color}色的{object_label}的{direction}侧"
        
        return StandardFormat(
            object_label=object_label,
            object_color=object_color,
            direction=direction,
            description=description
        )
    
    def parse_to_standard_format(self, text: str) -> Tuple[StandardFormat, float, str]:
        """
        智能解析自然语言到Text2Loc标准格式
        
        这是核心方法，通过Qwen模型分析理解自然语言，
        转换为Text2Loc标准格式，然后传递给原始Text2Loc系统
        
        Args:
            text: 自然语言描述
            
        Returns:
            (标准格式, 置信度, 意图)
        """
        start_time = time.time()
        
        try:
            # 调用Qwen模型进行智能分析
            if self.config.mock_mode:
                result = self._mock_parse(text)
            else:
                result = self._api_parse(text)
            
            parse_time = time.time() - start_time
            
            # 提取组件
            components = result.get("components", {})
            confidence = result.get("confidence", 0.0)
            
            # 映射到标准格式
            standard_format = self._map_to_standard_format(components)
            
            # 检测意图
            intent = self._detect_intent(text)
            
            logger.info(f"✅ 智能解析完成: {text}")
            logger.info(f"   标准格式: {standard_format.to_dict()}")
            logger.info(f"   置信度: {confidence:.2f}")
            logger.info(f"   意图: {intent}")
            
            return standard_format, confidence, intent
            
        except Exception as e:
            logger.error(f"智能解析失败: {e}")
            # 返回默认值
            default_format = StandardFormat(
                object_label="building",
                object_color="gray",
                direction="north",
                description="默认位置"
            )
            return default_format, 0.5, "unknown"
    



def test_example():
    """测试示例：我在树林靠近山的位置"""
    print("=" * 60)
    print("测试示例查询")
    print("=" * 60)
    
    # 创建引擎（模拟模式用于快速测试）
    config = NLUConfig(mock_mode=True, enable_dialog=True)
    engine = OptimizedNLUEngine(config)
    
    test_queries = [
        "我在树林靠近山的位置",
        "我站在红色大楼的北侧约5米处",
        "交通灯的东边有一个停车区域",
        "在桥的左侧，有一棵大树",
        "房子的前面有一棵树",
        "山脚下的小路旁边",
        "停车场的东北角",
        "湖的西边有一座桥",
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        result = engine.parse(query)
        
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  意图: {result.intent}")
        print(f"  解析结果:")
        
        for key, value in result.components.items():
            if isinstance(value, dict):
                print(f"    - {key}: {value.get('value')} (置信度: {value.get('confidence', 0):.2f})")
        
        if result.need_clarification:
            print(f"  需要澄清: {result.clarification_question}")
        
        print()


def test_interactive():
    """测试交互式查询"""
    print("\n" + "=" * 60)
    print("测试交互式查询（多轮对话）")
    print("=" * 60)
    
    config = NLUConfig(mock_mode=True, enable_dialog=True)
    engine = OptimizedNLUEngine(config)
    
    # 模拟多轮对话
    session_id = "test_session_001"
    
    print("\n【第1轮】")
    print("用户: 你好")
    result1 = engine.interactive_query("你好", session_id)
    print(f"助手: {result1.get('message', '你好！')}")
    
    print("\n【第2轮】")
    print("用户: 我在树林靠近山的位置")
    result2 = engine.interactive_query("我在树林靠近山的位置", session_id)
    print(f"解析结果: 置信度={result2['confidence']:.2f}")
    print(f"对象: {result2['components'].get('object', {}).get('value')}")
    
    print("\n【第3轮】")
    print("用户: 还有一棵大树")
    result3 = engine.interactive_query("还有一棵大树", session_id)
    print(f"解析结果: 置信度={result3['confidence']:.2f}")
    print(f"对象: {result3['components'].get('object', {}).get('value')}")
    
    print("\n【第4轮】")
    print("用户: 我站在红色大楼的北侧")
    result4 = engine.interactive_query("我站在红色大楼的北侧", session_id)
    print(f"解析结果: 置信度={result4['confidence']:.2f}")
    print(f"对象: {result4['components'].get('object', {}).get('value')}")
    print(f"方向: {result4['components'].get('direction', {}).get('value')}")
    print(f"颜色: {result4['components'].get('color', {}).get('value')}")
    
    # 查看对话历史
    context = engine.get_dialog_context(session_id)
    if context:
        print(f"\n对话历史记录: {len(context.history)} 轮")


def test_clarification():
    """测试澄清机制"""
    print("\n" + "=" * 60)
    print("测试澄清机制")
    print("=" * 60)
    
    config = NLUConfig(mock_mode=True, enable_dialog=True, confidence_threshold=0.7)
    engine = OptimizedNLUEngine(config)
    
    session_id = "test_session_002"
    
    print("\n用户: 在那里")
    result = engine.interactive_query("在那里", session_id)
    print(f"置信度: {result['confidence']:.2f}")
    print(f"需要澄清: {result['need_clarification']}")
    if result['need_clarification']:
        print(f"澄清问题: {result['clarification_question']}")
        print(f"建议: {result.get('suggestions', [])}")


if __name__ == "__main__":
    test_example()
    test_interactive()
    test_clarification()
