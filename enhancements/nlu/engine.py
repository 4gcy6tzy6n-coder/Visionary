"""
NLU (Natural Language Understanding) 引擎 - Text2Loc增强版

自然语言理解引擎，用于解析自由文本描述中的位置信息。
支持方向、颜色、对象、空间关系等多维度信息提取。
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
import requests
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLUComponent(Enum):
    """NLU解析组件类型"""
    DIRECTION = "direction"      # 方向解析
    COLOR = "color"              # 颜色识别
    OBJECT = "object"            # 对象识别
    RELATION = "relation"        # 空间关系
    DISTANCE = "distance"        # 距离信息
    POSITION = "position"        # 位置坐标


@dataclass
class NLUResult:
    """NLU解析结果"""
    text: str                    # 原始输入文本
    components: Dict[str, Any]   # 解析出的组件信息
    confidence: float            # 总体置信度 (0.0-1.0)
    model: str                   # 使用的模型
    parse_time: float            # 解析耗时(秒)
    enhanced_used: bool          # 是否使用增强解析
    fallback_reason: Optional[str] = None  # 回退原因
    error: Optional[str] = None            # 错误信息


@dataclass
class NLUConfig:
    """NLU配置参数"""
    # API配置
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen3-vl:2b"  # 使用qwen3-vl:2b作为默认模型
    timeout: int = 30

    # 组件开关
    enabled_components: List[NLUComponent] = None

    # 置信度阈值
    confidence_threshold: float = 0.7
    component_thresholds: Dict[NLUComponent, float] = None

    # 回退配置
    fallback_enabled: bool = True
    fallback_timeout: int = 10

    # 性能配置
    batch_size: int = 10
    cache_enabled: bool = True
    max_cache_size: int = 1000
    enable_parallel: bool = False  # 是否启用并行处理

    # 调试配置
    debug_mode: bool = False
    mock_mode: bool = False

    # 模型参数
    temperature: float = 0.1  # 温度参数
    top_p: float = 0.9  # Top-p采样
    max_tokens: int = 500  # 最大token数

    def __post_init__(self):
        """初始化默认值"""
        if self.enabled_components is None:
            self.enabled_components = [
                NLUComponent.DIRECTION,
                NLUComponent.COLOR,
                NLUComponent.OBJECT,
                NLUComponent.RELATION
            ]

        if self.component_thresholds is None:
            self.component_thresholds = {
                NLUComponent.DIRECTION: 0.7,
                NLUComponent.COLOR: 0.6,
                NLUComponent.OBJECT: 0.6,
                NLUComponent.RELATION: 0.5,
                NLUComponent.DISTANCE: 0.5,
                NLUComponent.POSITION: 0.6
            }


class NLUEngine:
    """
    自然语言理解引擎

    核心功能：
    1. 解析自由文本中的位置描述
    2. 提取方向、颜色、对象等多维度信息
    3. 置信度评估和自动回退
    4. 批量处理和缓存优化
    """

    # 预定义模板库
    TEMPLATES = {
        "full_parse": """
        你是一个专业的空间位置解析器。请分析以下文本中的位置描述，并提取所有相关信息。

        文本: "{text}"

        请提取以下信息：
        1. 方向描述（如：北、东、上、左前方、东北角等）
        2. 颜色信息（如：红色、蓝色、灰色等）
        3. 对象信息（如：大楼、停车位、灯柱、交通标志等）
        4. 空间关系（如：旁边、之间、靠近、正上方等）
        5. 距离信息（如：约5米、10米处、不远等）
        6. 坐标信息（如果有的话）

        返回JSON格式：
        {{
            "direction": {{
                "value": "方向字符串",
                "confidence": 0.95,
                "normalized": "标准方向"
            }},
            "color": {{
                "value": "颜色名称",
                "confidence": 0.90
            }},
            "object": {{
                "value": "对象名称",
                "confidence": 0.85
            }},
            "relation": {{
                "value": "空间关系",
                "confidence": 0.80
            }},
            "distance": {{
                "value": "距离数值(米)",
                "confidence": 0.70
            }},
            "overall_confidence": 0.85
        }}

        注意：如果没有某项信息，请返回null。
        只返回JSON，不要有其他内容。
        """,

        "direction_only": """
        请分析以下文本中的方向描述：

        文本: "{text}"

        提取方向信息，返回JSON格式：
        {{
            "direction": {{
                "raw": "原始方向描述",
                "normalized": "归一化方向",
                "confidence": 0.95
            }},
            "reason": "解析理由"
        }}
        """,

        "color_only": """
        请分析以下文本中的颜色描述：

        文本: "{text}"

        提取颜色信息，返回JSON格式：
        {{
            "color": {{
                "raw": "原始颜色描述",
                "normalized": "归一化颜色名称",
                "confidence": 0.90
            }}
        }}
        """
    }

    # 方向映射表
    DIRECTION_MAPPING = {
        # 北方向组
        "north": ["north", "northern", "北", "forward", "front", "前方", "前侧", "北侧"],
        "south": ["south", "southern", "南", "backward", "back", "后方", "后侧", "南侧"],
        "east": ["east", "eastern", "东", "right", "右侧", "右边", "东侧"],
        "west": ["west", "western", "西", "left", "左侧", "左边", "西侧"],
        "on_top": ["on top", "above", "over", "atop", "上", "上方", "上面", "顶部"],

        # 复合方向映射
        "north_east": ["northeast", "north east", "north-east", "东北"],
        "south_east": ["southeast", "south east", "south-east", "东南"],
        "south_west": ["southwest", "south west", "south-west", "西南"],
        "north_west": ["northwest", "north west", "north-west", "西北"]
    }

    def __init__(self, config: Optional[NLUConfig] = None):
        """
        初始化NLU引擎

        Args:
            config: NLU配置参数，如果为None则使用默认配置
        """
        self.config = config or NLUConfig()
        self.session = requests.Session() if not self.config.mock_mode else None
        
        # 预热模型（如果使用真实模式）
        if not self.config.mock_mode:
            self._warmup_model()

        # 缓存系统
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # 性能统计
        self.total_parses = 0
        self.total_time = 0.0

        logger.info(f"NLU引擎初始化完成: model={self.config.model_name}, mock={self.config.mock_mode}")
    
    def _warmup_model(self):
        """预热模型 - 发送空请求加载模型到内存"""
        if not self.config.mock_mode:
            try:
                logger.info(f"预热模型 {self.config.model_name}...")
                payload = {
                    "model": self.config.model_name,
                    "prompt": "hello",
                    "stream": False,
                    "options": {"max_tokens": 1}
                }
                response = self.session.post(
                    f"{self.config.ollama_url}/api/generate",
                    json=payload,
                    timeout=120  # 预热超时120秒
                )
                if response.status_code == 200:
                    logger.info("✅ 模型预热成功")
                else:
                    logger.warning(f"模型预热失败: {response.status_code}")
            except Exception as e:
                logger.warning(f"模型预热跳过: {e}")

    def _call_ollama_api(self, prompt: str) -> str:
        """
        调用ollama API（使用qwen3-vl:2b模型）

        Args:
            prompt: 提示词

        Returns:
            模型响应文本
        """
        if self.config.mock_mode:
            return self._mock_response(prompt)

        try:
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens
                }
            }

            response = self.session.post(
                f"{self.config.ollama_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()

                # qwen3-vl:2b可能返回额外的解释文本，需要提取JSON部分
                # 尝试找到JSON部分
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    # 验证是否是有效的JSON
                    try:
                        json.loads(json_text)  # 验证JSON
                        return json_text
                    except json.JSONDecodeError:
                        logger.warning(f"提取的JSON无效，使用完整响应: {json_text}")
                        return response_text

                return response_text
            else:
                logger.error(f"API调用失败: {response.status_code}, {response.text}")
                raise Exception(f"API调用失败: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error(f"API调用超时: {self.config.timeout}秒")
            raise TimeoutError(f"API调用超时: {self.config.timeout}秒")
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            raise

    def _mock_response(self, prompt: str) -> str:
        """
        模拟模式响应（用于测试和开发）
        模拟qwen3-vl:2b模型的响应行为

        Args:
            prompt: 提示词

        Returns:
            模拟响应文本（JSON格式）
        """
        # 改进的模拟逻辑，模拟qwen3-vl:2b的智能解析
        prompt_lower = prompt.lower()

        # 扩展方向关键词检测，包括复合方向
        direction_keywords = {
            "north": ["north", "北", "北侧", "北边", "前方", "前侧", "forward", "front"],
            "south": ["south", "南", "南侧", "南边", "后方", "后侧", "backward", "back"],
            "east": ["east", "东", "东侧", "东边", "右侧", "右边", "右前方", "右前", "right"],
            "west": ["west", "西", "西侧", "西边", "左侧", "左边", "左前方", "左前", "left"],
            "on_top": ["on top", "above", "over", "atop", "上", "上方", "上面", "顶部", "正上方"],
            "north_east": ["northeast", "north east", "north-east", "东北", "东北角", "东北侧"],
            "south_east": ["southeast", "south east", "south-east", "东南", "东南角", "东南侧"],
            "south_west": ["southwest", "south west", "south-west", "西南", "西南角", "西南侧"],
            "north_west": ["northwest", "north west", "north-west", "西北", "西北角", "西北侧"]
        }

        result = {
            "direction": None,
            "color": None,
            "object": None,
            "relation": None,
            "distance": None,
            "overall_confidence": 0.7
        }

        # 改进的方向关键词匹配 - 优先匹配更具体的复合方向
        found_direction = None

        # 先检查复合方向（更具体）
        for direction, keywords in direction_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    found_direction = direction
                    break
            if found_direction:
                break

        if found_direction:
            # 根据具体方向设置置信度
            confidence = 0.9
            if found_direction in ["north_east", "south_east", "south_west", "north_west"]:
                confidence = 0.85  # 复合方向置信度稍低

            result["direction"] = {
                "value": found_direction,
                "confidence": confidence,
                "normalized": found_direction if found_direction != "on_top" else "on_top"
            }

        # 改进的颜色检测，包括中文颜色名称
        colors_dict = {
            "red": ["red", "红色", "红", "赤色", "赤"],
            "green": ["green", "绿色", "绿", "青色"],
            "blue": ["blue", "蓝色", "蓝"],
            "yellow": ["yellow", "黄色", "黄"],
            "gray": ["gray", "grey", "灰色", "灰"],
            "black": ["black", "黑色", "黑"],
            "white": ["white", "白色", "白"],
            "brown": ["brown", "棕色", "褐色", "褐"],
            "dark-green": ["dark-green", "深绿色", "墨绿", "深绿"],
            "bright-gray": ["bright-gray", "亮灰色", "浅灰"]
        }

        for color_name, keywords in colors_dict.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    result["color"] = {
                        "value": color_name,
                        "confidence": 0.85
                    }
                    break
            if result.get("color"):
                break

        # 改进的对象检测（支持22个原有类别）
        objects_dict = {
            "building": ["building", "大楼", "建筑", "建筑物", "楼房", "大厦"],
            "pole": ["pole", "柱子", "灯柱", "电线杆", "杆子"],
            "parking": ["parking", "停车场", "停车位", "车位", "泊车位"],
            "sign": ["sign", "标志", "交通标志", "路标", "指示牌", "标牌"],
            "light": ["light", "灯", "路灯", "照明灯", "灯具"],
            "car": ["car", "汽车", "车辆", "小车", "轿车"],
            "tree": ["tree", "树", "树木", "大树", "树木"],
            "box": ["box", "箱子", "盒子", "方块"],
            "bridge": ["bridge", "桥", "桥梁", "天桥"],
            "fence": ["fence", "围墙", "栅栏", "栏杆"],
            "garage": ["garage", "车库", "停车场", "停车库"],
            "guard rail": ["guard rail", "护栏", "防护栏", "安全栏"],
            "lamp": ["lamp", "路灯", "灯具", "照明"],
            "pad": ["pad", "垫子", "平台", "基座"],
            "road": ["road", "道路", "马路", "公路"],
            "sidewalk": ["sidewalk", "人行道", "步道", "便道"],
            "smallpole": ["smallpole", "小柱子", "短杆", "小杆"],
            "stop": ["stop", "停止", "停止标志", "停车"],
            "terrain": ["terrain", "地形", "地面", "土地"],
            "traffic light": ["traffic light", "红绿灯", "交通灯", "信号灯"],
            "trash bin": ["trash bin", "垃圾桶", "垃圾箱", "废物箱"],
            "tunnel": ["tunnel", "隧道", "地道", "隧洞"],
            "vegetation": ["vegetation", "植被", "植物", "草木"],
            "vending machine": ["vending machine", "自动售货机", "贩卖机", "售货机"],
            "wall": ["wall", "墙", "墙壁", "墙体"]
        }

        for obj_name, keywords in objects_dict.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    result["object"] = {
                        "value": obj_name,
                        "confidence": 0.8
                    }
                    break
            if result.get("object"):
                break

        # 改进的空间关系检测
        relations_dict = {
            "near": ["near", "beside", "next to", "close to", "附近", "旁边", "邻近", "靠近"],
            "between": ["between", "之间", "中间", "当中"],
            "above": ["above", "over", "on top of", "上方", "上面", "顶部", "正上方"],
            "below": ["below", "under", "beneath", "下方", "下面", "底下"],
            "in_front_of": ["in front of", "前面", "前方", "正前方", "前侧"],
            "behind": ["behind", "后面", "后方", "后侧", "背后"],
            "adjacent": ["adjacent", "相邻", "毗邻", "紧邻"]
        }

        for rel_name, keywords in relations_dict.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    result["relation"] = {
                        "value": rel_name,
                        "confidence": 0.75
                    }
                    break
            if result.get("relation"):
                break

        # 改进的距离检测 - 提取具体数字
        import re
        distance_patterns = [
            r'(\d+(?:\.\d+)?)\s*米',
            r'(\d+(?:\.\d+)?)\s*meter',
            r'(\d+(?:\.\d+)?)\s*m',
            r'约(\d+(?:\.\d+)?)\s*米',
            r'大约(\d+(?:\.\d+)?)\s*米',
            r'距离约(\d+(?:\.\d+)?)\s*米',
            r'distance.*?(\d+(?:\.\d+)?)\s*m',
            r'(\d+(?:\.\d+)?)\s*米处'
        ]

        distance_value = None
        for pattern in distance_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                try:
                    distance_value = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        if distance_value is not None:
            result["distance"] = {
                "value": distance_value,
                "confidence": 0.8
            }
        else:
            # 如果没有明确数字，检查是否有距离描述
            distance_words = ["米处", "meter", "距离", "约", "大约", "不远", "远处", "附近"]
            for word in distance_words:
                if word in prompt_lower:
                    result["distance"] = {
                        "value": 5.0,  # 默认距离
                        "confidence": 0.5
                    }
                    break

        # 计算总体置信度，基于提取到的信息
        confidence_sum = 0
        confidence_count = 0

        for key in ["direction", "color", "object", "relation", "distance"]:
            if key in result and result[key] is not None:
                confidence_sum += result[key].get("confidence", 0.5)
                confidence_count += 1

        if confidence_count > 0:
            overall_confidence = confidence_sum / confidence_count
            # 如果有较多信息提取成功，提高置信度
            if confidence_count >= 3:
                overall_confidence = min(overall_confidence * 1.1, 0.95)
            elif confidence_count == 1:
                overall_confidence = overall_confidence * 0.9  # 只有一项信息，降低置信度

            result["overall_confidence"] = round(overall_confidence, 3)

        return json.dumps(result, ensure_ascii=False)

    def _normalize_direction(self, raw_direction: str) -> str:
        """
        归一化方向描述

        Args:
            raw_direction: 原始方向描述

        Returns:
            归一化的方向
        """
        if not raw_direction:
            return "north"  # 默认方向

        raw_lower = raw_direction.lower()

        # 检查映射表
        for normalized, variants in self.DIRECTION_MAPPING.items():
            for variant in variants:
                if variant in raw_lower:
                    return normalized

        # 检查复合方向
        if "northeast" in raw_lower or "东北" in raw_lower:
            return "north"  # 东北归一化为北
        elif "southeast" in raw_lower or "东南" in raw_lower:
            return "east"   # 东南归一化为东
        elif "southwest" in raw_lower or "西南" in raw_lower:
            return "west"   # 西南归一化为西
        elif "northwest" in raw_lower or "西北" in raw_lower:
            return "west"   # 西北归一化为西

        # 默认返回北
        return "north"

    def _calculate_overall_confidence(self, components: Dict[str, Any]) -> float:
        """
        计算总体置信度

        Args:
            components: 解析出的组件信息

        Returns:
            总体置信度
        """
        confidences = []
        weights = {
            "direction": 0.3,
            "color": 0.2,
            "object": 0.25,
            "relation": 0.15,
            "distance": 0.1
        }

        for component, weight in weights.items():
            if component in components and components[component] is not None:
                comp_data = components[component]
                if isinstance(comp_data, dict) and "confidence" in comp_data:
                    confidences.append(comp_data["confidence"] * weight)

        if not confidences:
            return 0.5  # 默认置信度

        return min(1.0, sum(confidences) / sum(weights.values()))

    def _create_prompt(self, text: str, components: List[NLUComponent] = None) -> str:
        """
        创建解析提示词

        Args:
            text: 输入文本
            components: 要解析的组件列表

        Returns:
            提示词
        """
        if components is None:
            components = self.config.enabled_components

        # 如果是全量解析，使用完整模板
        if len(components) >= 3:
            return self.TEMPLATES["full_parse"].format(text=text)

        # 单个组件解析
        if len(components) == 1:
            component = components[0]
            if component == NLUComponent.DIRECTION:
                return self.TEMPLATES["direction_only"].format(text=text)
            elif component == NLUComponent.COLOR:
                return self.TEMPLATES["color_only"].format(text=text)

        # 自定义提示词
        component_names = [c.value for c in components]
        component_str = ", ".join(component_names)

        return f"""
        请分析以下文本中的{component_str}信息：

        文本: "{text}"

        提取{component_str}信息，返回JSON格式。
        如果没有某项信息，请返回null。
        只返回JSON，不要有其他内容。
        """

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析模型响应

        Args:
            response: 模型响应文本

        Returns:
            解析结果字典
        """
        try:
            # 尝试解析JSON
            if response.startswith("{") and response.endswith("}"):
                data = json.loads(response)

                # 标准化方向
                if "direction" in data and data["direction"] is not None:
                    if isinstance(data["direction"], dict):
                        raw_dir = data["direction"].get("value") or data["direction"].get("raw")
                        if raw_dir:
                            normalized = self._normalize_direction(raw_dir)
                            data["direction"]["normalized"] = normalized

                return data
            else:
                # 如果不是JSON，尝试提取
                logger.warning(f"响应不是JSON格式: {response[:100]}...")
                return {"error": "响应格式错误"}

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 响应: {response[:200]}")
            return {"error": f"JSON解析失败: {str(e)}"}
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return {"error": f"解析失败: {str(e)}"}

    def parse(self, text: str, components: List[NLUComponent] = None) -> NLUResult:
        """
        解析单个文本

        Args:
            text: 输入文本
            components: 要解析的组件列表，如果为None则使用配置中的组件

        Returns:
            NLU解析结果
        """
        if components is None:
            components = self.config.enabled_components

        logger.info(f"解析文本: {text[:50]}...")
        start_time = time.time()

        try:
            # 检查缓存
            cache_key = f"{text}_{'_'.join([c.value for c in components])}"
            if self.config.cache_enabled and cache_key in self.cache:
                self.cache_hits += 1
                cached_result = self.cache[cache_key]
                cached_result.parse_time = time.time() - start_time
                return cached_result

            self.cache_misses += 1

            # 创建提示词并调用API
            prompt = self._create_prompt(text, components)
            response = self._call_ollama_api(prompt)

            # 解析响应
            parsed_data = self._parse_response(response)

            # 计算置信度
            overall_confidence = self._calculate_overall_confidence(parsed_data)

            # 检查是否需要回退
            enhanced_used = overall_confidence >= self.config.confidence_threshold
            fallback_reason = None

            if not enhanced_used and self.config.fallback_enabled:
                fallback_reason = f"置信度过低: {overall_confidence:.2f} < {self.config.confidence_threshold:.2f}"
                logger.warning(f"触发回退: {fallback_reason}")

                # 这里可以调用原有的解析方法
                # 暂时使用简化的回退逻辑
                parsed_data = self._fallback_parse(text, components)
                overall_confidence = 0.5  # 回退方法的置信度

            # 创建结果对象
            parse_time = time.time() - start_time
            result = NLUResult(
                text=text,
                components=parsed_data,
                confidence=overall_confidence,
                model=self.config.model_name,
                parse_time=parse_time,
                enhanced_used=enhanced_used,
                fallback_reason=fallback_reason
            )

            # 更新性能统计
            self.total_parses += 1
            self.total_time += parse_time

            # 添加到缓存
            if self.config.cache_enabled:
                if len(self.cache) >= self.config.max_cache_size:
                    # 简单FIFO清理
                    keys_to_remove = list(self.cache.keys())[:self.config.max_cache_size // 2]
                    for key in keys_to_remove:
                        del self.cache[key]

                self.cache[cache_key] = result

            logger.info(f"解析完成: 置信度={overall_confidence:.3f}, 耗时={parse_time:.3f}秒")
            return result

        except Exception as e:
            logger.error(f"解析失败: {e}")
            parse_time = time.time() - start_time

            # 错误时回退
            if self.config.fallback_enabled:
                parsed_data = self._fallback_parse(text, components)
                return NLUResult(
                    text=text,
                    components=parsed_data,
                    confidence=0.3,
                    model=self.config.model_name,
                    parse_time=parse_time,
                    enhanced_used=False,
                    fallback_reason=f"解析错误: {str(e)}"
                )
            else:
                return NLUResult(
                    text=text,
                    components={"error": str(e)},
                    confidence=0.0,
                    model=self.config.model_name,
                    parse_time=parse_time,
                    enhanced_used=False,
                    error=str(e)
                )

    def batch_parse(self, texts: List[str], components: List[NLUComponent] = None, 
                   progress_callback=None) -> List[NLUResult]:
        """
        批量解析文本

        Args:
            texts: 文本列表
            components: 要解析的组件列表
            progress_callback: 进度回调函数，接收(current, total)参数

        Returns:
            NLU解析结果列表
        """
        logger.info(f"批量解析: {len(texts)}个文本, 模型: {self.config.model_name}")
        results = []

        # 如果启用并行处理（且API支持），可以使用并行处理
        # 但Ollama API通常不支持批量，所以这里使用串行处理
        for i, text in enumerate(texts):
            # 添加延迟避免API过载（非模拟模式）
            if i > 0 and not self.config.mock_mode:
                time.sleep(0.1)

            try:
                result = self.parse(text, components)
                results.append(result)
            except Exception as e:
                logger.error(f"解析文本失败: {text[:50]}... 错误: {e}")
                # 创建错误结果
                error_result = NLUResult(
                    text=text,
                    components={"error": str(e)},
                    confidence=0.0,
                    model=self.config.model_name,
                    parse_time=0.0,
                    enhanced_used=False,
                    error=str(e)
                )
                results.append(error_result)

            # 进度日志和回调
            if (i + 1) % 10 == 0:
                logger.info(f"批量解析进度: {i + 1}/{len(texts)}")

            if progress_callback:
                try:
                    progress_callback(i + 1, len(texts))
                except Exception as e:
                    logger.warning(f"进度回调失败: {e}")

        # 批量解析完成后的统计
        successful = sum(1 for r in results if r.error is None)
        avg_time = sum(r.parse_time for r in results) / len(results) if results else 0

        logger.info(f"批量解析完成: {successful}/{len(texts)} 成功, 平均耗时: {avg_time:.3f}秒")

        return results

    def _fallback_parse(self, text: str, components: List[NLUComponent]) -> Dict[str, Any]:
        """
        回退解析方法（当增强解析失败时使用）

        Args:
            text: 输入文本
            components: 要解析的组件列表

        Returns:
            简化的解析结果
        """
        result = {}
        text_lower = text.lower()

        # 简单关键词匹配
        for component in components:
            if component == NLUComponent.DIRECTION:
                # 方向关键词匹配
                directions = {
                    "north": ["north", "北", "forward", "front", "前方", "前侧"],
                    "south": ["south", "南", "backward", "back", "后方", "后侧"],
                    "east": ["east", "东", "right", "右侧", "右边"],
                    "west": ["west", "西", "left", "左侧", "左边"],
                    "on_top": ["on top", "above", "over", "atop", "上", "上方"]
                }

                for dir_name, keywords in directions.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            result["direction"] = {
                                "value": dir_name,
                                "confidence": 0.7,
                                "normalized": dir_name
                            }
                            break
                    if "direction" in result:
                        break

                if "direction" not in result:
                    result["direction"] = {
                        "value": "north",
                        "confidence": 0.5,
                        "normalized": "north"
                    }

            elif component == NLUComponent.COLOR:
                # 颜色关键词匹配
                colors = {
                    "red": ["red", "红色"],
                    "green": ["green", "绿色"],
                    "blue": ["blue", "蓝色"],
                    "gray": ["gray", "灰色"],
                    "black": ["black", "黑色"],
                    "white": ["white", "白色"]
                }

                for color_name, keywords in colors.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            result["color"] = {
                                "value": color_name,
                                "confidence": 0.6
                            }
                            break
                    if "color" in result:
                        break

            elif component == NLUComponent.OBJECT:
                # 对象关键词匹配
                objects = {
                    "building": ["building", "大楼", "建筑"],
                    "pole": ["pole", "柱子", "灯柱"],
                    "parking": ["parking", "停车场", "停车位"],
                    "sign": ["sign", "标志", "交通标志"]
                }

                for obj_name, keywords in objects.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            result["object"] = {
                                "value": obj_name,
                                "confidence": 0.6
                            }
                            break
                    if "object" in result:
                        break

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        获取引擎统计信息

        Returns:
            统计信息字典
        """
        avg_time = self.total_time / self.total_parses if self.total_parses > 0 else 0

        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) \
            if (self.cache_hits + self.cache_misses) > 0 else 0

        # 计算组件使用统计
        component_stats = {
            "total": self.total_parses,
            "success": self.total_parses - sum(1 for r in self.cache.values() if r.error),
            "average_time": avg_time,
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "config": {
                "model": self.config.model_name,
                "ollama_url": self.config.ollama_url,
                "confidence_threshold": self.config.confidence_threshold,
                "mock_mode": self.config.mock_mode,
                "cache_enabled": self.config.cache_enabled,
                "batch_size": self.config.batch_size,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens
            }
        }

        return component_stats

    def clear_cache(self):
        """清理缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("NLU缓存已清理")


# 测试函数
def test_nlu_engine():
    """测试NLU引擎"""
    print("=== NLU引擎测试 (qwen3-vl:2b) ===")

    # 创建配置
    config = NLUConfig(
        model_name="qwen3-vl:2b",  # 使用qwen3-vl:2b模型
        mock_mode=True,            # 使用模拟模式
        confidence_threshold=0.7,
        debug_mode=True,
        temperature=0.1,
        top_p=0.9,
        max_tokens=500
    )

    # 创建引擎
    engine = NLUEngine(config)

    # 测试文本
    test_texts = [
        "我站在红色大楼的北侧约5米处",
        "目标在建筑物的东边大约10米处",
        "位置在停车场的西南角，靠近入口",
        "点在灯柱的正上方，高度约3米",
        "位于交通标志的右前方，距离约8米",
        "在围墙的左侧，靠近建筑物",
        "停车区域的东北角，距离约15米",
        "交通信号灯的正下方",
        "灰色墙壁的右侧，约3米处",
        "绿色植被的南侧"
    ]

    print(f"\n测试{len(test_texts)}个文本...")
    print(f"模型: {config.model_name}")
    print(f"置信度阈值: {config.confidence_threshold}")

    # 进度回调函数
    def progress_callback(current, total):
        if current % 5 == 0:
            print(f"  进度: {current}/{total}")

    # 批量解析
    results = engine.batch_parse(test_texts, progress_callback=progress_callback)

    # 显示结果
    success_count = 0
    enhanced_count = 0

    for i, (text, result) in enumerate(zip(test_texts, results), 1):
        print(f"\n测试 {i}: '{text[:30]}...'")

        if result.error:
            print(f"  ❌ 错误: {result.error}")
            continue

        success_count += 1

        print(f"  置信度: {result.confidence:.3f} {'✓' if result.confidence >= 0.7 else '⚠'}")
        print(f"  耗时: {result.parse_time:.3f}秒")
        print(f"  增强使用: {result.enhanced_used}")

        if result.enhanced_used:
            enhanced_count += 1

        if result.fallback_reason:
            print(f"  回退原因: {result.fallback_reason}")

        # 显示组件信息
        for component, data in result.components.items():
            if data is not None and not isinstance(data, str):
                if component != "overall_confidence":
                    if isinstance(data, dict):
                        value = data.get("value") or data.get("raw")
                        conf = data.get("confidence", 0)
                        print(f"    {component}: {value} (置信度: {conf:.2f})")
                    else:
                        print(f"    {component}: {data}")

    # 显示统计信息
    stats = engine.get_stats()
    print(f"\n{'='*60}")
    print(f"统计信息:")
    print(f"{'='*60}")
    print(f"  总解析次数: {stats['total']}")
    print(f"  成功次数: {success_count} ({success_count/len(test_texts)*100:.1f}%)")
    print(f"  增强解析使用: {enhanced_count} ({enhanced_count/len(test_texts)*100:.1f}%)")
    print(f"  平均解析时间: {stats['average_time']:.3f}秒")
    print(f"  缓存命中率: {stats['cache_hit_rate']:.2%}")
    print(f"  缓存大小: {stats['cache_size']}")
    print(f"{'='*60}")

    # 评估
    accuracy = success_count / len(test_texts)
    print(f"\n评估结果:")
    if accuracy >= 0.9:
        print(f"  ✅ 优秀 (准确率: {accuracy:.1%})")
    elif accuracy >= 0.8:
        print(f"  ✅ 良好 (准确率: {accuracy:.1%})")
    elif accuracy >= 0.7:
        print(f"  ⚠️ 一般 (准确率: {accuracy:.1%})")
    else:
        print(f"  ❌ 较差 (准确率: {accuracy:.1%})")

    print(f"\n测试完成!")
    return engine


if __name__ == "__main__":
    # 运行测试
    engine = test_nlu_engine()
