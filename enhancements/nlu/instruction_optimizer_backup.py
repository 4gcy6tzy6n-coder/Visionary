"""
InstructionOptimizer - 基于Qwen模型的智能指令优化器

负责理解用户意图、优化查询、智能补全、标准化术语
生成最终的优化查询传递给模板生成层
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import requests
import hashlib
import logging

from enhancements.nlu.optimized_engine import OptimizedNLUEngine

logger = logging.getLogger(__name__)


@dataclass
class OptimizedQuery:
    """优化后的查询结果"""

    original_input: str
    optimized_input: str
    parsed_elements: Dict[str, Any]
    confidence_scores: Dict[str, float]
    optimization_log: List[str]
    need_clarification: bool
    clarification_types: List[str]
    suggested_clarifications: List[str]
    query_id: str
    timestamp: str
    session_context: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "original_input": self.original_input,
            "optimized_input": self.optimized_input,
            "parsed_elements": self.parsed_elements,
            "confidence_scores": self.confidence_scores,
            "optimization_log": self.optimization_log,
            "need_clarification": self.need_clarification,
            "clarification_types": self.clarification_types,
            "suggested_clarifications": self.suggested_clarifications,
            "session_context": self.session_context
        }


@dataclass
class ClarificationIntent:
    """澄清意图"""

    issue_type: str  # ambiguous_object, missing_direction, vague_relation
    description: str
    confidence: float
    priority: int  # 1-5, higher = more urgent
    candidates: List[str]  # possible values/options
    severity: str  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "issue_type": self.issue_type,
            "description": self.description,
            "confidence": self.confidence,
            "priority": self.priority,
            "candidates": self.candidates,
            "severity": self.severity
        }


@dataclass
class ClarificationNeed:
    """澄清需求描述"""

    issue_type: str  # ambiguous_object, missing_direction, vague_relation, etc.
    description: str
    confidence: float
    priority: int  # 1-5, higher = more urgent
    candidates: List[str]  # possible values/options


class InstructionOptimizer:
    """基于Qwen模型的智能指令优化器"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen3-vl:2b",
        mock_mode: bool = True,
        cache_enabled: bool = True,
        cache_size: int = 1000
    ):
        """
        初始化指令优化器

        Args:
            ollama_url: Ollama API地址
            model_name: Qwen模型名称
            mock_mode: 是否使用模拟模式
            cache_enabled: 是否启用缓存
            cache_size: 缓存大小
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.mock_mode = mock_mode
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size

        # 初始化NLU引擎用于解析
        self.nlu_engine = OptimizedNLUEngine()

        # 查询缓存
        self.cache: Dict[str, OptimizedQuery] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # 性能统计
        self.total_queries = 0
        self.total_time = 0.0
        self.total_clarifications = 0

        # 会话上下文管理
        self.session_contexts: Dict[str, Dict] = {}

        # 优化策略配置
        self.optimization_strategies = [
            "clarify_ambiguity",
            "complete_missing",
            "normalize_terms",
            "generate_variants",
            "assess_confidence"
        ]

        # 补全字典
        self.completion_dict = {
            "orientation": ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "on-top"],
            "color": ["red", "green", "blue", "yellow", "gray", "black", "white", "brown", "orange", "purple"],
            "object": ["building", "pole", "parking", "sign", "light", "car", "tree", "box", "bridge", "fence",
                      "road", "sidewalk", "wall", "vegetation", "grass", "water", "rock", "path", "entrance", "corner"],
            "relation": ["near", "between", "above", "below", "in_front_of", "behind", "left_of", "right_of", "at", "facing"],
        }

        logger.info(f"InstructionOptimizer 初始化完成: mock={mock_mode}, model={model_name}")

    def _generate_cache_key(self, user_input: str, session_context: Optional[Dict] = None) -> str:
        """生成缓存键"""
        key_data = user_input
        if session_context:
            key_data += json.dumps(session_context, sort_keys=True)
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()

    def _get_session_context(self, session_id: Optional[str]) -> Optional[Dict]:
        """获取会话上下文"""
        if session_id and session_id in self.session_contexts:
            return self.session_contexts[session_id]
        return None

    def update_session_context(self, session_id: str, context_data: Dict):
        """更新会话上下文"""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {}
        self.session_contexts[session_id].update(context_data)

    def _call_qwen_analysis(self, user_input: str) -> Dict[str, Any]:
        """
        调用Qwen模型进行深度分析

        Args:
            user_input: 用户输入的自然语言

        Returns:
            模型分析结果
        """
        if self.mock_mode:
            return self._mock_qwen_analysis(user_input)

        try:
            prompt = self._create_qwen_prompt(user_input)

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                return self._parse_qwen_response(response_text)
            else:
                logger.error(f"Qwen API调用失败: {response.status_code}")
                return self._mock_qwen_analysis(user_input)

        except Exception as e:
            logger.error(f"Qwen API调用异常: {e}")
            return self._mock_qwen_analysis(user_input)

    def _mock_qwen_analysis(self, user_input: str) -> Dict[str, Any]:
        """模拟Qwen分析结果"""

        # 解析基本要素
        parsed = self.nlu_engine.parse(user_input)

        # 提取各组件
        components = parsed.components if parsed.components else {}

        # 检测模糊元素
        ambiguities = []
        missing_elements = []

        # 检查方向
        if not components.get("direction"):
            missing_elements.append("direction")
            ambiguities.append(ClarificationNeed(
                issue_type="missing_direction",
                description="缺少方向信息",
                confidence=0.9,
                priority=3,
                candidates=["north", "south", "east", "west", "on-top"]
            ))
        elif components["direction"].get("confidence", 0) < 0.6:
            ambiguities.append(ClarificationNeed(
                issue_type="ambiguous_direction",
                description="方向信息模糊",
                confidence=components["direction"]["confidence"],
                priority=2,
                candidates=["north", "south", "east", "west"]
            ))

        # 检查对象
        if not components.get("object"):
            missing_elements.append("object")
            ambiguities.append(ClarificationNeed(
                issue_type="missing_object",
                description="缺少对象信息",
                confidence=0.9,
                priority=2,
                candidates=["building", "pole", "parking", "sign", "light"]
            ))
        elif components["object"].get("confidence", 0) < 0.6:
            ambiguities.append(ClarificationNeed(
                issue_type="ambiguous_object",
                description="对象识别模糊",
                confidence=components["object"]["confidence"],
                priority=3,
                candidates=["building", "pole", "car", "tree"]
            ))

        # 检查颜色
        if not components.get("color"):
            missing_elements.append("color")
        elif components["color"].get("confidence", 0) < 0.6:
            ambiguities.append(ClarificationNeed(
                issue_type="ambiguous_color",
                description="颜色识别模糊",
                confidence=components["color"]["confidence"],
                priority=1,
                candidates=["red", "green", "blue", "gray", "black"]
            ))

        # 生成优化建议
        optimization_suggestions = []
        if missing_elements:
            optimization_suggestions.append(f"补全缺失信息: {', '.join(missing_elements)}")
        if ambiguities:
            optimization_suggestions.append(f"澄清模糊元素: {len(ambiguities)}个")

        return {
            "entities": {
                "direction": components.get("direction", {}),
                "color": components.get("color", {}),
                "object": components.get("object", {}),
                "relation": components.get("relation", {}),
                "distance": components.get("distance", {})
            },
            "intent": self._detect_intent(user_input),
            "confidence": parsed.confidence if hasattr(parsed, 'confidence') else 0.7,
            "ambiguities": [
                {
                    "issue_type": a.issue_type,
                    "description": a.description,
                    "confidence": a.confidence,
                    "priority": a.priority,
                    "candidates": a.candidates
                }
                for a in ambiguities
            ],
            "missing_elements": missing_elements,
            "optimization_suggestions": optimization_suggestions,
            "needs_more_info": len(missing_elements) > 0 or len(ambiguities) > 0
        }

    def _create_qwen_prompt(self, user_input: str) -> str:
        """创建Qwen分析提示词"""

        prompt = f"""你是一个专业的空间定位查询分析系统。请分析以下自然语言查询，并返回JSON格式的结果：

用户查询: "{user_input}"

请分析以下要素：
1. 方向 (direction): 上下左右、前后等方向信息
2. 颜色 (color): 物体的颜色描述
3. 对象 (object): 需要定位的物体类别
4. 关系 (relation): 空间关系（靠近、中间、上方等）
5. 距离 (distance): 距离信息

返回JSON格式：
{{
  "entities": {{
    "direction": {{"value": "具体值或null", "confidence": 0.0-1.0}},
    "color": {{"value": "具体值或null", "confidence": 0.0-1.0}},
    "object": {{"value": "具体值或null", "confidence": 0.0-1.0}},
    "relation": {{"value": "具体值或null", "confidence": 0.0-1.0}},
    "distance": {{"value": "具体值或null", "confidence": 0.0-1.0}}
  }},
  "intent": "具体意图类型",
  "confidence": 0.0-1.0,
  "needs_clarification": true/false,
  "clarification_needs": ["missing_direction", "ambiguous_object"],
  "optimization_suggestions": ["补全方向信息", "澄清对象类型"]
}}

请确保：
1. 返回纯JSON格式，不包含额外文本
2. confidence值为0.0-1.0之间
3. null值用于缺失信息
4. needs_clarification根据置信度和完整性判断"""

        return prompt

    def _parse_qwen_response(self, response_text: str) -> Dict[str, Any]:
        """解析Qwen响应"""
        try:
            # 提取JSON部分
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return self._mock_qwen_analysis("解析失败")
        except Exception as e:
            logger.error(f"解析Qwen响应失败: {e}")
            return self._mock_qwen_analysis("解析异常")

    def _detect_intent(self, text: str) -> str:
        """检测用户意图"""
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ["怎么走", "方向", "朝向", "面向", "direction"]):
            return "ask_direction"
        elif any(keyword in text_lower for keyword in ["在哪里", "位置", "地方", "方位", "locate", "find"]):
            return "query_location"
        elif any(keyword in text_lower for keyword in ["我在", "位于", "处于", "站在", "坐在"]):
            return "describe_location"
        elif any(keyword in text_lower for keyword in ["附近", "旁边", "靠近", "nearby"]):
            return "find_nearby"
        elif any(keyword in text_lower for keyword in ["哪个", "什么", "哪里", "which", "what"]):
            return "clarify"
        elif any(keyword in text_lower for keyword in ["你好", "hello", "hi", "您好"]):
            return "greeting"
        else:
            return "unknown"

    def _complete_missing_info(self, analysis: Dict[str, Any]) -> str:
        """补全缺失信息"""
        original = self._extract_original_query(analysis)
        missing = analysis.get("missing_elements", [])

        if not missing:
            return original

        # 根据上下文智能补全
        completed_query = original

        # 简单补全策略
        completion_map = {
            "direction": "当前位置",
            "color": "默认颜色",
            "object": "主要目标",
            "relation": "附近",
            "distance": "一定距离"
        }

        for missing_elem in missing:
            if missing_elem in completion_map:
                completion = completion_map[missing_elem]
                if missing_elem == "direction":
                    completed_query = f"{completion}的{completed_query}"
                elif missing_elem == "object":
                    completed_query = f"{completed_query}的{completion}"

        return completed_query

    def _normalize_terms(self, query: str) -> str:
        """标准化术语"""
        # 同义词映射
        synonyms = {
            "前方": "north",
            "北侧": "north",
            "右边": "east",
            "东边": "east",
            "左边": "west",
            "西边": "west",
            "后面": "south",
            "南侧": "south",
            "上方": "on-top",
            "顶部": "on-top",
            "红色": "red",
            "绿色": "green",
            "蓝色": "blue",
            "大楼": "building",
            "灯柱": "pole",
            "停车场": "parking",
            "路标": "sign",
        }

        normalized = query
        for zh, en in synonyms.items():
            if zh in normalized:
                normalized = normalized.replace(zh, en)

        return normalized

    def _assess_confidence_level(self, analysis: Dict[str, Any]) -> float:
        """评估整体置信度"""
        entities = analysis.get("entities", {})

        confidences = []
        for entity_type, entity_data in entities.items():
            if entity_data and "confidence" in entity_data:
                confidences.append(entity_data["confidence"])

        if not confidences:
            return 0.5

        # 加权平均，方向和对象权重更高
        weights = {"direction": 1.5, "object": 1.5, "color": 1.0, "relation": 1.0, "distance": 0.5}
        weighted_sum = 0
        total_weight = 0

        for entity_type, entity_data in entities.items():
            if entity_data and "confidence" in entity_data:
                weight = weights.get(entity_type, 1.0)
                weighted_sum += entity_data["confidence"] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _extract_original_query(self, analysis: Dict[str, Any]) -> str:
        """从分析中提取原始查询"""
        # 这里可以根据分析结果重构查询
        # 简化实现：使用entities构建
        entities = analysis.get("entities", {})
        parts = []

        if entities.get("direction") and entities["direction"].get("value"):
            parts.append(entities["direction"]["value"])
        if entities.get("color") and entities["color"].get("value"):
            parts.append(entities["color"]["value"])
        if entities.get("object") and entities["object"].get("value"):
            parts.append(entities["object"]["value"])

        return " ".join(parts) if parts else "unknown"

    def _needs_clarification(self, analysis: Dict[str, Any]) -> bool:
        """判断是否需要澄清"""
        needs = analysis.get("needs_clarification", False)
        confidence = analysis.get("confidence", 0.5)

        # 如果低置信度或缺失关键信息，需要澄清
        if confidence < 0.7:
            return True

        if needs:
            return True

        # 检查歧义
        ambiguities = analysis.get("ambiguities", [])
        if len(ambiguities) > 0:
            # 如果有高优先级的歧义，需要澄清
            high_priority = [a for a in ambiguities if a.get("priority", 0) >= 3]
            if len(high_priority) > 0:
                return True

        return False

    def _generate_clarification_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """生成澄清问题"""
        questions = []

        missing = analysis.get("missing_elements", [])
        ambiguities = analysis.get("ambiguities", [])

        # 基于缺失信息的问题
        if "direction" in missing:
            questions.append("请问是哪个方向？（如：北、南、东、西、上）")
        if "object" in missing:
            questions.append("请问是哪个物体？（如：大楼、柱子、汽车、树木）")
        if "color" in missing:
            questions.append("请问是什么颜色？（如：红、绿、蓝、灰）")

        # 基于歧义的问题
        for amb in ambiguities:
            if amb.get("issue_type") == "ambiguous_object":
                candidates = amb.get("candidates", [])
                if candidates:
                    question = f"您说的可能是哪个物体？（如：{'、'.join(candidates[:3])}）"
                    questions.append(question)

        if not questions:
            # 默认澄清问题
            questions.append("能说得更具体一些吗？")

        return questions

    def _generate_suggested_responses(self) -> List[str]:
        """生成建议的用户响应"""
        return [
            "北侧的红色大楼",
            "停车场附近的绿色树木",
            "东边的交通标志",
            "上方的路灯",
            "左侧的建筑物"
        ]

    def _should_generate_variants(self, analysis: Dict[str, Any]) -> bool:
        """判断是否需要生成变体"""
        confidence = analysis.get("confidence", 0.5)

        # 低置信度或有歧义时生成变体
        if confidence < 0.8:
            return True

        ambiguities = analysis.get("ambiguities", [])
        if len(ambiguities) > 0:
            return True

        return False

    def _generate_query_variants(self, analysis: Dict[str, Any]) -> List[str]:
        """生成查询变体"""
        variants = []

        entities = analysis.get("entities", {})

        # 从解析结果中提取可能的选项
        if entities.get("direction"):
            dir_value = entities["direction"].get("value")
            if dir_value:
                variants.append(f"{dir_value}的物体")
                variants.append(f"物体在{dir_value}侧")

        if entities.get("object"):
            obj_value = entities["object"].get("value")
            if obj_value:
                if entities.get("direction"):
                    dir_value = entities["direction"].get("value", "北")
                    variants.append(f"{dir_value}侧的{obj_value}")
                variants.append(f"{obj_value}位置")

        if entities.get("color") and entities.get("object"):
            color_value = entities["color"].get("value")
            obj_value = entities["object"].get("value")
            if color_value and obj_value:
                variants.append(f"{color_value}的{obj_value}")
                if entities.get("direction"):
                    dir_value = entities["direction"].get("value", "北")
                    variants.append(f"{dir_value}侧的{color_value}{obj_value}")

        # 保持至少2个变体
        if len(variants) < 2:
            variants.append("当前场景的主要物体")
            variants.append("当前位置附近的物体")

        return variants[:5]  # 最多5个

    def optimize(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        session_context: Optional[Dict] = None
    ) -> OptimizedQuery:
        """
        核心优化流程：
        1. 缓存检查
        2. Qwen深度理解分析
        3. 智能补全缺失信息
        4. 标准化术语
        5. 评估置信度
        6. 判断是否需要澄清
        7. 生成优化查询

        Args:
            user_input: 用户输入
            session_id: 会话ID
            session_context: 会话上下文

        Returns:
            优化后的查询
        """
        start_time = time.time()

        # 缓存检查
        cache_key = self._generate_cache_key(user_input, session_context or self._get_session_context(session_id))
        if self.cache_enabled and cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            cached_result.query_id = f"cache_{cache_key[:8]}"
            return cached_result

        self.cache_misses += 1
        self.total_queries += 1

        try:
            # 步骤1: Qwen深度理解分析
            qwen_analysis = self._call_qwen_analysis(user_input)

            # 步骤2: 智能补全缺失信息
            completed_query = self._complete_missing_info(qwen_analysis)

            # 步骤3: 标准化术语
            normalized_query = self._normalize_terms(completed_query)

            # 步骤4: 评估置信度
            confidence = self._assess_confidence_level(qwen_analysis)

            # 步骤5: 判断是否需要澄清
            need_clarification = self._needs_clarification(qwen_analysis)

            # 步骤6: 生成澄清问题和建议
            clarification_questions = []
            suggested_clarifications = []

            if need_clarification:
                clarification_questions = self._generate_clarification_questions(qwen_analysis)
                suggested_clarifications = self._generate_suggested_responses()
                self.total_clarifications += 1

            # 步骤7: 生成查询变体（如果需要）
            query_variants = []
            if self._should_generate_variants(qwen_analysis):
                query_variants = self._generate_query_variants(qwen_analysis)

            # 生成优化日志
            optimization_log = [
                f"Qwen分析完成，置信度: {confidence:.2f}",
                f"检测到{len(qwen_analysis.get('missing_elements', []))}个缺失元素",
                f"检测到{len(qwen_analysis.get('ambiguities', []))}个歧义元素",
                f"补全查询: {completed_query}",
                f"标准化查询: {normalized_query}",
            ]

            if need_clarification:
                optimization_log.append(f"需要澄清，生成{len(clarification_questions)}个问题")

            if query_variants:
                optimization_log.append(f"生成{len(query_variants)}个查询变体")

            # 提取解析元素
            parsed_elements = qwen_analysis.get("entities", {})

            # 提取置信度分数
            confidence_scores = {
                "direction": parsed_elements.get("direction", {}).get("confidence", 0.0),
                "color": parsed_elements.get("color", {}).get("confidence", 0.0),
                "object": parsed_elements.get("object", {}).get("confidence", 0.0),
                "relation": parsed_elements.get("relation", {}).get("confidence", 0.0),
                "distance": parsed_elements.get("distance", {}).get("confidence", 0.0),
                "overall": confidence
            }

            # 澄清类型
            clarification_types = []
            if need_clarification:
                amb = qwen_analysis.get("ambiguities", [])
                clarification_types = [a.get("issue_type") for a in amb]

            # 构建结果
            query_id = f"opt_{int(time.time())}_{self.total_queries}"
            timestamp = datetime.now().isoformat()

            result = OptimizedQuery(
                original_input=user_input,
                optimized_input=normalized_query,
                parsed_elements=parsed_elements,
                confidence_scores=confidence_scores,
                optimization_log=optimization_log,
                need_clarification=need_clarification,
                clarification_types=clarification_types,
                suggested_clarifications=suggested_clarifications,
                query_id=query_id,
                timestamp=timestamp,
                session_context=session_context
            )

            # 缓存结果
            if self.cache_enabled:
                if len(self.cache) >= self.cache_size:
                    # 移除最旧的缓存
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[cache_key] = result

            elapsed = time.time() - start_time
            self.total_time += elapsed

            logger.info(f"✅ 指令优化完成: {elapsed*1000:.1f}ms")
            logger.info(f"   原始: {user_input}")
            logger.info(f"   优化: {normalized_query}")
            logger.info(f"   置信度: {confidence:.2f}")
            logger.info(f"   需要澄清: {need_clarification}")

            return result

        except Exception as e:
            logger.error(f"指令优化失败: {e}")

            # 返回默认优化结果
            return OptimizedQuery(
                original_input=user_input,
                optimized_input=user_input,
                parsed_elements={},
                confidence_scores={"overall": 0.5},
                optimization_log=[f"优化失败: {str(e)}"],
                need_clarification=True,
                clarification_types=["unknown"],
                suggested_clarifications=["请更具体地描述"],
                query_id=f"error_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                session_context=session_context
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_time = self.total_time / self.total_queries if self.total_queries > 0 else 0.0

        return {
            "total_queries": self.total_queries,
            "total_time_ms": self.total_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.cache),
            "total_clarifications": self.total_clarifications,
            "mock_mode": self.mock_mode,
            "model_name": self.model_name
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("缓存已清空")


@dataclass
class ClarificationResponse:
    """澄清响应"""

    need_clarification: bool
    questions: List[str]
    clarification_type: str
    session_state: Dict
    suggested_responses: List[str]


@dataclass
class ClarificationResult:
    """澄清结果"""

    success: bool
    updated_query: Optional[OptimizedQuery]
    clarification_complete: bool
    next_steps: str


@dataclass
class ClarificationSession:
    """澄清会话"""

    session_id: str
    original_query: str
    clarification_history: List[Dict]
    current_question: Optional['ClarificationQuestion']
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def update(self):
        """更新时间戳"""
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "clarification_history": self.clarification_history,
            "current_question": self.current_question.to_dict() if self.current_question else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "clarification_count": len(self.clarification_history)
        }


@dataclass
class ClarificationQuestion:
    """澄清问题"""

    question_id: str
    question_text: str
    issue_type: str
    priority: int
    options: Optional[List[str]] = None
    suggested_answer: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "issue_type": self.issue_type,
            "priority": self.priority,
            "options": self.options,
            "suggested_answer": self.suggested_answer,
            "timestamp": self.timestamp
        }


class InteractiveClarifierManager:
    """交互式澄清管理器"""

    def __init__(self, instruction_optimizer: InstructionOptimizer):
        """
        初始化澄清管理器

        Args:
            instruction_optimizer: 指令优化器实例
        """
        self.optimizer = instruction_optimizer
        self.active_sessions: Dict[str, ClarificationSession] = {}

        logger.info("InteractiveClarifierManager 初始化完成")

    def create_clarification_session(self, session_id: str, initial_query: str) -> ClarificationSession:
        """创建澄清会话"""
        session = ClarificationSession(
            session_id=session_id,
            original_query=initial_query,
            clarification_history=[],
            current_question=None
        )
        self.active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ClarificationSession]:
        """获取会话"""
        return self.active_sessions.get(session_id)

    def end_session(self, session_id: str):
        """结束会话"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    def process_response(self, session_id: str, user_response: str) -> ClarificationResult:
        """处理用户响应"""
        session = self.get_session(session_id)
        if not session:
            return ClarificationResult(
                success=False,
                updated_query=None,
                clarification_complete=False,
                next_steps="session_not_found"
            )

        # 记录响应
        session.clarification_history.append({
            "question": session.current_question,
            "response": user_response,
            "timestamp": datetime.now().isoformat()
        })

        # 重新优化查询
        updated_query = self.optimizer.optimize(
            user_input=user_response,
            session_id=session_id
        )

        # 更新会话
        session.current_question = None

        # 检查是否完成澄清
        clarification_complete = not updated_query.need_clarification

        return ClarificationResult(
            success=True,
            updated_query=updated_query,
            clarification_complete=clarification_complete,
            next_steps="proceed_to_retrieval" if clarification_complete else "need_more_clarification"
        )


@dataclass
class ClarificationSession:
    """澄清会话"""

    session_id: str
    original_query: str
    clarification_history: List[Dict]
    current_question: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "clarification_history": self.clarification_history,
            "current_question": self.current_question,
            "clarification_count": len(self.clarification_history)
        }


# 测试函数
def test_instruction_optimizer():
    """测试指令优化器"""
    print("=" * 60)
    print("测试 InstructionOptimizer")
    print("=" * 60)

    optimizer = InstructionOptimizer(mock_mode=True)

    test_cases = [
        "停车场附近的红色汽车",
        "建筑物旁边的灯柱",
        "我要找绿色的东西",
        "哪里有大屏显示器",
        "北侧的白色车辆",
    ]

    for query in test_cases:
        print(f"\n测试查询: {query}")
        result = optimizer.optimize(query)

        print(f"  优化后: {result.optimized_input}")
        print(f"  置信度: {result.confidence_scores.get('overall', 0):.2f}")
        print(f"  需要澄清: {result.need_clarification}")
        print(f"  解析要素: {list(result.parsed_elements.keys())}")

        if result.need_clarification:
            print(f"  澄清问题: {result.suggested_clarifications}")

    # 打印统计信息
    print(f"\n{'='*60}")
    print("性能统计")
    print(f"{'='*60}")
    stats = optimizer.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def test_clarification_workflow():
    """测试澄清流程"""
    print("\n" + "=" * 60)
    print("测试澄清流程")
    print("=" * 60)

    optimizer = InstructionOptimizer(mock_mode=True)
    clarifier = InteractiveClarifierManager(optimizer)

    # 创建会话
    session_id = "test_session_001"
    initial_query = "找一个东西"

    print(f"\n初始查询: {initial_query}")

    # 第一次优化（应该需要澄清）
    result1 = optimizer.optimize(initial_query, session_id=session_id)
    print(f"第一次优化，需要澄清: {result1.need_clarification}")

    if result1.need_clarification:
        # 创建澄清会话
        session = clarifier.create_clarification_session(session_id, initial_query)
        print(f"创建澄清会话: {session.session_id}")

        # 模拟用户响应
        user_response = "停车场附近的红色汽车"
        print(f"\n用户响应: {user_response}")

        # 处理响应
        result2 = clarifier.process_response(session_id, user_response)
        print(f"澄清完成: {result2.clarification_complete}")

        if result2.updated_query:
            print(f"更新后查询: {result2.updated_query.optimized_input}")

    # 清理会话
    clarifier.end_session(session_id)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    test_instruction_optimizer()
    test_clarification_workflow()
