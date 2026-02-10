"""
InteractiveClarifier - 交互式澄清器

负责处理模糊查询的交互式澄清系统
支持多轮对话、澄清问题生成、会话状态管理
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClarificationQuestion:
    """澄清问题"""

    question_id: str
    question_text: str
    issue_type: str  # ambiguous_object, missing_direction, vague_relation, etc.
    priority: int  # 1-5
    options: Optional[List[str]] = None  # 可选答案
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


@dataclass
class ClarificationSession:
    """澄清会话"""

    session_id: str
    original_query: str
    clarification_history: List[Dict]
    current_question: Optional[ClarificationQuestion]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # active, completed, expired

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
            "status": self.status,
            "clarification_count": len(self.clarification_history),
            "is_active": self.status == "active"
        }


@dataclass
class ClarificationIntent:
    """澄清意图"""

    issue_type: str
    description: str
    confidence: float
    priority: int
    candidates: List[str]
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
class UserResponseParsed:
    """用户响应解析结果"""

    resolved_issues: List[str]
    new_entities: Dict[str, str]
    confidence: float
    ambiguity_detected: bool
    additional_clarification_needed: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "resolved_issues": self.resolved_issues,
            "new_entities": self.new_entities,
            "confidence": self.confidence,
            "ambiguity_detected": self.ambiguity_detected,
            "additional_clarification_needed": self.additional_clarification_needed
        }


class InteractiveClarifier:
    """交互式澄清系统"""

    # 澄清策略映射
    CLARIFICATION_STRATEGIES = {
        "ambiguous_object": "_clarify_ambiguous_object",
        "missing_direction": "_clarify_missing_direction",
        "missing_color": "_clarify_missing_color",
        "vague_relation": "_clarify_vague_relation",
        "ambiguous_direction": "_clarify_ambiguous_direction",
        "multiple_options": "_clarify_multiple_options",
        "unclear_intent": "_clarify_unclear_intent",
    }

    # 澄清问题模板
    QUESTION_TEMPLATES = {
        "ambiguous_object": {
            "en": "What object are you referring to?",
            "zh": "您指的是哪个物体？"
        },
        "missing_direction": {
            "en": "Which direction are you looking at?",
            "zh": "您看的是哪个方向？"
        },
        "missing_color": {
            "en": "What color is the object?",
            "zh": "物体是什么颜色的？"
        },
        "vague_relation": {
            "en": "How is this related to the reference point?",
            "zh": "这与参考点有什么关系？"
        },
        "ambiguous_direction": {
            "en": "Which direction exactly? (north/south/east/west/on-top)",
            "zh": "具体是哪个方向？（北/南/东/西/上）"
        },
        "multiple_options": {
            "en": "Which option best matches your query?",
            "zh": "哪个选项最符合您的查询？"
        },
        "unclear_intent": {
            "en": "Could you provide more details about what you're looking for?",
            "zh": "能提供更多关于您要找什么的细节吗？"
        }
    }

    # 建议回答选项
    SUGGESTED_RESPONSES = {
        "missing_direction": ["北侧", "南侧", "东侧", "西侧", "上方"],
        "ambiguous_object": ["大楼", "柱子", "汽车", "树木", "墙", "灯"],
        "missing_color": ["红色", "绿色", "蓝色", "灰色", "黑色", "白色"],
        "vague_relation": ["很近", "附近", "旁边", "中间", "前方"],
        "ambiguous_direction": ["北", "南", "东", "西", "上"]
    }

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen3-vl:2b",
        mock_mode: bool = True,
        max_clarification_count: int = 5,
        session_timeout: int = 300,
        language: str = "zh"  # "zh" or "en"
    ):
        """
        初始化交互式澄清器

        Args:
            ollama_url: Ollama API地址
            model_name: Qwen模型名称
            mock_mode: 是否使用模拟模式
            max_clarification_count: 最大澄清次数
            session_timeout: 会话超时时间（秒）
            language: 默认语言
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.mock_mode = mock_mode
        self.max_clarification_count = max_clarification_count
        self.session_timeout = session_timeout
        self.language = language

        # 会话管理
        self.sessions: Dict[str, ClarificationSession] = {}
        self.session_counter = 0

        # 澄清意图统计
        self.total_clarifications = 0
        self.total_confirmed = 0
        self.total_failed = 0
        self.total_time = 0.0

        # 缓存最近的问题，避免重复
        self.recent_questions: Dict[str, str] = {}

        logger.info(f"InteractiveClarifier 初始化完成: mock={mock_mode}, language={language}")

    def create_session(
        self,
        session_id: Optional[str] = None,
        original_query: str = "",
        initial_clarifications: Optional[List[ClarificationIntent]] = None
    ) -> ClarificationSession:
        """
        创建新的澄清会话

        Args:
            session_id: 会话ID（可选，自动生成）
            original_query: 原始查询
            initial_clarifications: 初始澄清意图列表

        Returns:
            澄清会话
        """
        if session_id is None:
            session_id = f"clarify_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.session_counter}"
            self.session_counter += 1

        session = ClarificationSession(
            session_id=session_id,
            original_query=original_query,
            clarification_history=[],
            current_question=None,
            status="active"
        )

        self.sessions[session_id] = session

        # 如果有初始澄清意图，生成第一个问题
        if initial_clarifications:
            sorted_clarifications = sorted(initial_clarifications, key=lambda x: x.priority, reverse=True)
            first_clarification = sorted_clarifications[0]
            question = self._generate_clarification_question(first_clarification, session)
            session.current_question = question
            session.update()

        logger.info(f"创建澄清会话: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ClarificationSession]:
        """获取会话"""
        session = self.sessions.get(session_id)

        # 检查超时
        if session:
            if self._is_session_expired(session):
                session.status = "expired"
                session.update()
                return session

        return session

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除澄清会话: {session_id}")
            return True
        return False

    def _is_session_expired(self, session: ClarificationSession) -> bool:
        """检查会话是否过期"""
        if session.status != "active":
            return True

        # 检查超时
        try:
            from datetime import datetime
            created_time = datetime.fromisoformat(session.created_at)
            elapsed = (datetime.now() - created_time).total_seconds()
            return elapsed > self.session_timeout
        except:
            return False

    def _clean_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = []
        for session_id, session in list(self.sessions.items()):
            if self._is_session_expired(session):
                session.status = "expired"
                session.update()
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            t = time.time() - 300  # 5分钟超时
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")

    def identify_clarification_needs(
        self,
        parsed_result: Any,
        original_query: str,
        confidence_threshold: float = 0.7
    ) -> List[ClarificationIntent]:
        """
        识别需要澄清的方面

        Args:
            parsed_result: 解析结果
            original_query: 原始查询
            confidence_threshold: 置信度阈值

        Returns:
            澄清意图列表
        """
        clarifications = []

        try:
            # 提取解析结果
            if hasattr(parsed_result, 'to_dict'):
                result_dict = parsed_result.to_dict()
            else:
                result_dict = parsed_result

            entities = result_dict.get("entities", {})
            confidence_scores = result_dict.get("confidence_scores", {})
            overall_confidence = result_dict.get("confidence_scores", {}).get("overall", 0.5)

            # 1. 检查整体置信度
            if overall_confidence < confidence_threshold:
                clarifications.append(ClarificationIntent(
                    issue_type="unclear_intent",
                    description="查询不够明确",
                    confidence=1.0 - overall_confidence,
                    priority=4,
                    candidates=[],
                    severity="high"
                ))

            # 2. 检查缺失的方向
            direction = entities.get("direction", {}).get("value") if entities.get("direction") else None
            if not direction:
                direction_conf = confidence_scores.get("direction", 0.0)
                clarifications.append(ClarificationIntent(
                    issue_type="missing_direction",
                    description="缺少方向信息",
                    confidence=1.0 - direction_conf,
                    priority=3,
                    candidates=["北", "南", "东", "西", "上"],
                    severity="high" if entities.get("object") else "medium"
                ))
            elif confidence_scores.get("direction", 0.5) < 0.6:
                clarifications.append(ClarificationIntent(
                    issue_type="ambiguous_direction",
                    description="方向信息模糊",
                    confidence=1.0 - confidence_scores.get("direction", 0.5),
                    priority=2,
                    candidates=["北", "南", "东", "西", "上"],
                    severity="medium"
                ))

            # 3. 检查缺失的物体
            obj = entities.get("object", {}).get("value") if entities.get("object") else None
            if not obj:
                obj_conf = confidence_scores.get("object", 0.0)
                clarifications.append(ClarificationIntent(
                    issue_type="ambiguous_object",
                    description="缺少对象信息",
                    confidence=1.0 - obj_conf,
                    priority=2,
                    candidates=["大楼", "柱子", "汽车", "树木", "墙", "灯"],
                    severity="medium"
                ))
            elif confidence_scores.get("object", 0.5) < 0.6:
                clarifications.append(ClarificationIntent(
                    issue_type="ambiguous_object",
                    description="对象识别模糊",
                    confidence=1.0 - confidence_scores.get("object", 0.5),
                    priority=3,
                    candidates=["大楼", "柱子", "汽车", "树木", "墙", "灯"],
                    severity="high"
                ))

            # 4. 检查缺失的颜色
            color = entities.get("color", {}).get("value") if entities.get("color") else None
            if not color and obj:  # 如果有物体但没有颜色，可以省略颜色
                pass  # 颜色是可选的
            elif color and confidence_scores.get("color", 0.5) < 0.6:
                clarifications.append(ClarificationIntent(
                    issue_type="ambiguous_color",
                    description="颜色识别模糊",
                    confidence=1.0 - confidence_scores.get("color", 0.5),
                    priority=1,
                    candidates=["红色", "绿色", "蓝色", "灰色", "黑色", "白色"],
                    severity="low"
                ))

            # 5. 检查关系
            relation = entities.get("relation", {}).get("value") if entities.get("relation") else None
            if relation and confidence_scores.get("relation", 0.5) < 0.6:
                clarifications.append(ClarificationIntent(
                    issue_type="vague_relation",
                    description="关系信息模糊",
                    confidence=1.0 - confidence_scores.get("relation", 0.5),
                    priority=1,
                    candidates=["很近", "附近", "旁边", "前面", "后面"],
                    severity="low"
                ))

            # 6. 检查最近的问题（避免重复）
            recent_issues = set()
            for session in self.sessions.values():
                if session.current_question:
                    recent_issues.add(session.current_question.issue_type)

            clarifications = [c for c in clarifications if c.issue_type not in recent_issues]

            # 按优先级排序
            clarifications.sort(key=lambda x: x.priority, reverse=True)

            logger.info(f"识别到 {len(clarifications)} 个澄清需求")

        except Exception as e:
            logger.error(f"识别澄清需求失败: {e}")
            # 返回默认的澄清需求
            clarifications.append(ClarificationIntent(
                issue_type="unclear_intent",
                description="查询不够明确",
                confidence=0.8,
                priority=4,
                candidates=[],
                severity="high"
            ))

        return clarifications

    def generate_clarification_questions(
        self,
        clarification_intents: List[ClarificationIntent],
        session: ClarificationSession,
        max_questions: int = 3,
        language: Optional[str] = None
    ) -> List[ClarificationQuestion]:
        """
        生成澄清问题

        Args:
            clarification_intents: 澄清意图列表
            session: 澄清会话
            max_questions: 最大问题数
            language: 语言（可选，默认使用初始化语言）

        Returns:
            澄清问题列表
        """
        if language is None:
            language = self.language

        questions = []
        used_issue_types = set()

        for intent in clarification_intents:
            if len(questions) >= max_questions:
                break

            if intent.issue_type in used_issue_types:
                continue

            # 避免近期已问过的问题
            if self._is_issue_recently_asked(intent.issue_type, session.session_id):
                continue

            question = self._generate_clarification_question(intent, session, language)
            if question:
                questions.append(question)
                used_issue_types.add(intent.issue_type)

        return questions

    def _generate_clarification_question(
        self,
        clarification_intent: ClarificationIntent,
        session: ClarificationSession,
        language: Optional[str] = None
    ) -> Optional[ClarificationQuestion]:
        """
        生成单个澄清问题

        Args:
            clarification_intent: 澄清意图
            session: 澄清会话
            language: 语言

        Returns:
            澄清问题
        """
        if language is None:
            language = self.language

        # 获取问题模板
        template_dict = self.QUESTION_TEMPLATES.get(clarification_intent.issue_type)
        if not template_dict:
            logger.warning(f"未知的澄清问题类型: {clarification_intent.issue_type}")
            return None

        # 选择语言
        if language in template_dict:
            question_text = template_dict[language]
        elif "en" in template_dict:
            question_text = template_dict["en"]
        elif "zh" in template_dict:
            question_text = template_dict["zh"]
        else:
            question_text = "请提供更多信息"

        # 从澄清结果中提取更多上下文信息（如果可用）
        if clarification_intent.candidates and len(clarification_intent.candidates) > 0:
            candidates_text = "、".join(clarification_intent.candidates[:3])
            if language == "zh":
                question_text = f"{question_text}（如：{candidates_text}）"
            else:
                question_text = f"{question_text} (e.g., {candidates_text})"

        # 生成建议回答
        suggested_answer = None
        if clarification_intent.issue_type in self.SUGGESTED_RESPONSES:
            candidates = self.SUGGESTED_RESPONSES[clarification_intent.issue_type]
            suggested_answer = candidates[0] if candidates else None

        # 生成选项
        options = None
        if clarification_intent.candidates and len(clarification_intent.candidates) > 0:
            options = clarification_intent.candidates[:5]  # 最多5个选项

        # 创建问题ID
        question_id = f"q_{clarification_intent.issue_type}_{session.session_id}_{len(session.clarification_history)}"

        # 更新会话
        question = ClarificationQuestion(
            question_id=question_id,
            question_text=question_text,
            issue_type=clarification_intent.issue_type,
            priority=clarification_intent.priority,
            options=options,
            suggested_answer=suggested_answer
        )

        session.current_question = question
        session.update()

        # 记录到最近问题缓存
        self.recent_questions[session.session_id] = clarification_intent.issue_type

        self.total_clarifications += 1

        return question

    def _is_issue_recently_asked(self, issue_type: str, session_id: str) -> bool:
        """检查问题类型是否最近已问过"""
        recent_issue = self.recent_questions.get(session_id)
        return recent_issue == issue_type

    def analyze_user_response(
        self,
        session_id: str,
        user_response: str,
        language: Optional[str] = None
    ) -> UserResponseParsed:
        """
        分析用户澄清响应

        Args:
            session_id: 会话ID
            user_response: 用户响应
            language: 语言

        Returns:
            解析结果
        """
        if language is None:
            language = self.language

        session = self.get_session(session_id)
        if not session or session.status != "active":
            logger.error(f"会话不存在或未激活: {session_id}")
            return UserResponseParsed(
                resolved_issues=[],
                new_entities={},
                confidence=0.0,
                ambiguity_detected=True,
                additional_clarification_needed=["session_invalid"]
            )

        current_question = session.current_question
        if not current_question:
            logger.error(f"会话 {session_id} 没有当前问题")
            return UserResponseParsed(
                resolved_issues=[],
                new_entities={},
                confidence=0.0,
                ambiguity_detected=True,
                additional_clarification_needed=["no_current_question"]
            )

        # 分析用户响应
        resolved_issues = []
        new_entities = {}
        ambiguity_detected = False
        additional_clarification_needed = []

        # 在模拟模式下使用简单的规则分析
        if self.mock_mode:
            return self._mock_analyze_user_response(
                session, user_response, current_question, language
            )

        # 使用Qwen模型分析（实际实现）
        try:
            # 这里可以调用Qwen模型进行更智能的分析
            pass
        except Exception as e:
            logger.error(f"分析用户响应失败: {e}")
            return self._mock_analyze_user_response(
                session, user_response, current_question, language
            )

    def _mock_analyze_user_response(
        self,
        session: ClarificationSession,
        user_response: str,
        current_question: ClarificationQuestion,
        language: str
    ) -> UserResponseParsed:
        """模拟分析用户响应"""
        resolved_issues = []
        new_entities = {}
        ambiguity_detected = False
        additional_clarification_needed = []

        response_lower = user_response.lower()

        # 根据当前问题类型分析
        issue_type = current_question.issue_type

        if issue_type == "missing_direction" or issue_type == "ambiguous_direction":
            # 解析方向
            direction_map = {
                "北": "north", "北方": "north", "北边": "north", "north": "north",
                "南": "south", "南方": "south", "南边": "south", "south": "south",
                "东": "east", "东方": "east", "东边": "east", "east": "east",
                "西": "west", "西方": "west", "西边": "west", "west": "west",
                "上": "on-top", "上方": "on-top", "上面": "on-top", "on-top": "on-top",
            }
            for zh, en in direction_map.items():
                if zh in response_lower or en in response_lower:
                    new_entities["direction"] = en
                    resolved_issues.append(issue_type)
                    break

        elif issue_type == "missing_object" or issue_type == "ambiguous_object":
            # 解析物体
            object_map = {
                "大楼": "building", "建筑": "building", "building": "building",
                "柱子": "pole", "灯柱": "pole", "pole": "pole",
                "汽车": "car", "车辆": "car", "car": "car",
                "树木": "tree", "树": "tree", "tree": "tree",
                "墙": "wall", "墙壁": "wall", "wall": "wall",
                "灯": "light", "路灯": "light", "light": "light",
                "停车场": "parking", "parking": "parking",
                "路标": "sign", "sign": "sign",
            }
            for zh, en in object_map.items():
                if zh in response_lower or en in response_lower:
                    new_entities["object"] = en
                    resolved_issues.append(issue_type)
                    break

        elif issue_type == "missing_color":
            # 解析颜色
            color_map = {
                "红": "red", "红色": "red", "red": "red",
                "绿": "green", "绿色": "green", "green": "green",
                "蓝": "blue", "蓝色": "blue", "blue": "blue",
                "灰": "gray", "灰色": "gray", "gray": "gray",
                "黑": "black", "黑色": "black", "black": "black",
                "白": "white", "白色": "white", "white": "white",
            }
            for zh, en in color_map.items():
                if zh in response_lower or en in response_lower:
                    new_entities["color"] = en
                    resolved_issues.append(issue_type)
                    break

        elif issue_type == "vague_relation":
            # 解析关系
            relation_map = {
                "很近": "near", "附近": "near", "near": "near",
                "旁边": "near", "beside": "near",
                "中间": "between", "之间": "between", "between": "between",
                "前面": "in_front_of", "前方": "in_front_of",
                "后面": "behind", "后方": "behind",
            }
            for zh, en in relation_map.items():
                if zh in response_lower or en in response_lower:
                    new_entities["relation"] = en
                    resolved_issues.append(issue_type)
                    break

        # 检查是否存在子问题
        if len(resolved_issues) == 0:
            # 用户响应不够明确
            ambiguity_detected = True
            if issue_type in ["missing_direction", "ambiguous_direction"]:
                additional_clarification_needed.append("方向不够明确")
            elif issue_type in ["missing_object", "ambiguous_object"]:
                additional_clarification_needed.append("物体不够明确")
            elif issue_type == "missing_color":
                additional_clarification_needed.append("颜色不够明确")

        # 计算置信度
        confidence_score = 0.5
        if resolved_issues:
            confidence_score = 0.9
        elif len(response_lower) > 10:
            confidence_score = 0.7

        result = UserResponseParsed(
            resolved_issues=resolved_issues,
            new_entities=new_entities,
            confidence=confidence_score,
            ambiguity_detected=ambiguity_detected,
            additional_clarification_needed=additional_clarification_needed
        )

        return result

    def process_clarification_response(
        self,
        session_id: str,
        user_response: str,
        language: Optional[str] = None,
        max_followup_questions: int = 1
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[ClarificationQuestion]]:
        """
        处理用户澄清响应，决定下一步行动

        Args:
            session_id: 会话ID
            user_response: 用户响应
            language: 语言
            max_followup_questions: 最大后续问题数

        Returns:
            (是否完成, 更新后的数据, 后续问题列表)
        """
        start_time = time.time()

        if language is None:
            language = self.language

        session = self.get_session(session_id)
        if not session or session.status != "active":
            logger.error(f"会话不存在或未激活: {session_id}")
            return False, None, []

        # 分析用户响应
        analysis = self.analyze_user_response(session_id, user_response, language)

        # 记录到历史
        session.clarification_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": session.current_question.to_dict() if session.current_question else None,
            "user_response": user_response,
            "analysis": analysis.to_dict()
        })

        # 更新会话
        session.current_question = None
        session.update()

        # 判断是否完成
        if analysis.resolved_issues:
            # 成功解析
            updated_data = {
                "resolved_issues": analysis.resolved_issues,
                "new_entities": analysis.new_entities,
                "confidence": analysis.confidence,
                "total_clarifications": len(session.clarification_history)
            }

            self.total_confirmed += 1
            elapsed = time.time() - start_time
            self.total_time += elapsed

            logger.info(f"✅ 澄清完成: {len(analysis.resolved_issues)}个问题已解决")

            return True, updated_data, []

        else:
            # 需要继续澄清
            if analysis.ambiguity_detected or len(session.clarification_history) >= self.max_clarification_count:
                # 过于模糊或达到最大澄清次数，终止会话
                session.status = "completed"
                session.update()

                self.total_failed += 1
                elapsed = time.time() - start_time
                self.total_time += elapsed

                logger.warning(f"⚠️ 澄清失败: 达到限制或过于模糊")

                return False, None, []

            else:
                # 生成后续问题
                followup_intents = [ClarificationIntent(
                    issue_type="unclear_intent",
                    description=analysis.additional_clarification_needed[0] if analysis.additional_clarification_needed else "需要更多信息",
                    confidence=0.7,
                    priority=3,
                    candidates=[],
                    severity="medium"
                )]

                followup_questions = self.generate_clarification_questions(
                    followup_intents, session, max_questions=1, language=language
                )

                elapsed = time.time() - start_time
                self.total_time += elapsed

                logger.info(f"✅ 需要继续澄清: 生成{len(followup_questions)}个后续问题")

                return False, None, followup_questions

    def batch_process_clarifications(
        self,
        clarification_intents: List[ClarificationIntent],
        original_query: str,
        language: Optional[str] = None
    ) -> Tuple[List[ClarificationQuestion], List[Dict[str, Any]]]:
        """
        批量处理澄清意图，生成问题和解析

        Args:
            clarification_intents: 澄清意图列表
            original_query: 原始查询
            language: 语言

        Returns:
            (问题列表, 解析结果列表)
        """
        if language is None:
            language = self.language

        # 创建临时会话
        temp_session = self.create_session(
            original_query=original_query,
            initial_clarifications=clarification_intents
        )

        questions = []
        if temp_session.current_question:
            questions.append(temp_session.current_question)

        # 为剩下的意图生成问题
        if len(clarification_intents) > 1:
            remaining_intents = clarification_intents[1:]
            additional_questions = self.generate_clarification_questions(
                remaining_intents, temp_session, language=language
            )
            questions.extend(additional_questions)

        # 我们只返回问题，解析需要用户响应后才能处理
        parsed_results = []

        # 删除临时会话
        self.delete_session(temp_session.session_id)

        return questions, parsed_results

    def generate_suggested_response(
        self,
        session_id: str,
        num_variants: int = 3
    ) -> List[str]:
        """
        生成建议的用户响应

        Args:
            session_id: 会话ID
            num_variants: 变体数量

        Returns:
            建议响应列表
        """
        session = self.get_session(session_id)
        if not session or not session.current_question:
            return []

        issue_type = session.current_question.issue_type

        # 从预定义选项中获取
        suggested_responses = self.SUGGESTED_RESPONSES.get(issue_type, [])
        if suggested_responses:
            return suggested_responses[:num_variants]

        # 默认建议
        return ["北侧", "南侧", "东侧", "西侧", "附近"]

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话状态

        Args:
            session_id: 会话ID

        Returns:
            会话状态信息
        """
        session = self.get_session(session_id)
        if not session:
            return {"status": "not_found"}

        return session.to_dict()

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """获取所有活跃会话"""
        return [
            session.to_dict()
            for session in self.sessions.values()
            if session.status == "active"
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_time = self.total_time / self.total_clarifications if self.total_clarifications > 0 else 0.0

        return {
            "total_clarifications": self.total_clarifications,
            "total_confirmed": self.total_confirmed,
            "total_failed": self.total_failed,
            "success_rate": self.total_confirmed / self.total_clarifications if self.total_clarifications > 0 else 0.0,
            "total_time_ms": self.total_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "active_sessions": len(self.sessions),
            "model_name": self.model_name,
            "mock_mode": self.mock_mode,
            "language": self.language
        }

    def clear_cache(self):
        """清空缓存"""
        self.recent_questions.clear()
        logger.info("缓存已清空")

    def export_sessions(self) -> Dict[str, Any]:
        """导出会话数据"""
        return {
            "sessions": {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            },
            "stats": self.get_stats()
        }

    def import_sessions(self, data: Dict[str, Any]):
        """导入会话数据"""
        if "sessions" in data:
            for session_id, session_data in data["sessions"].items():
                session = ClarificationSession(
                    session_id=session_data["session_id"],
                    original_query=session_data["original_query"],
                    clarification_history=session_data["clarification_history"],
                    current_question=None,
                    created_at=session_data["created_at"],
                    updated_at=session_data["updated_at"],
                    status=session_data["status"]
                )

                if session_data.get("current_question"):
                    question_data = session_data["current_question"]
                    session.current_question = ClarificationQuestion(
                        question_id=question_data["question_id"],
                        question_text=question_data["question_text"],
                        issue_type=question_data["issue_type"],
                        priority=question_data["priority"],
                        options=question_data.get("options"),
                        suggested_answer=question_data.get("suggested_answer"),
                        timestamp=question_data["timestamp"]
                    )

                self.sessions[session_id] = session

        logger.info(f"导入了 {len(self.sessions)} 个会话")


# ���试函数
def test_interactive_clarifier():
    """测试交互式澄清器"""
    print("=" * 60)
    print("测试 InteractiveClarifier")
    print("=" * 60)

    clarifier = InteractiveClarifier(mock_mode=True, language="zh")

    # 测试用例1：创建会话并生成问题
    print("\n测试用例1: 创建会话并生成问题")

    clarification_intents = [
        ClarificationIntent(
            issue_type="missing_direction",
            description="缺少方向信息",
            confidence=0.9,
            priority=3,
            candidates=["北", "南", "东", "西", "上"],
            severity="high"
        ),
        ClarificationIntent(
            issue_type="ambiguous_object",
            description="对象识别模糊",
            confidence=0.7,
            priority=2,
            candidates=["大楼", "柱子", "汽车", "树木"],
            severity="medium"
        )
    ]

    questions, _ = clarifier.batch_process_clarifications(
        clarification_intents,
        "找一个东西"
    )

    print(f"生成了 {len(questions)} 个澄清问题:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q.question_text}")
        print(f"     类型: {q.issue_type}, 优先级: {q.priority}")
        if q.suggested_answer:
            print(f"     建议回答: {q.suggested_answer}")

    # 测试用例2：完整交互流程
    print("\n测试用例2: 完整交互流程")

    # 创建会话
    session = clarifier.create_session(
        original_query="我要找一个东西",
        initial_clarifications=clarification_intents
    )

    print(f"创建会话: {session.session_id}")
    if session.current_question:
        print(f"当前问题: {session.current_question.issue_type} - {session.current_question.question_text}")

    # 模拟用户响应
    user_response1 = "北侧的红色汽车"
    print(f"\n用户响应: {user_response1}")

    completed, updated_data, followup_questions = clarifier.process_clarification_response(
        session.session_id, user_response1
    )

    print(f"是否完成: {completed}")
    if updated_data:
        print(f"更新数据: {updated_data}")
    if followup_questions:
        print(f"后续问题: {[q.question_text for q in followup_questions]}")

    # 检查会话状态
    status = clarifier.get_session_status(session.session_id)
    print(f"\n会话状态: {status['status']}")
    print(f"澄清次数: {status['clarification_count']}")

    # 测试用例3：解析直接响应
    print("\n测试用例3: 解析用户明确响应")

    session2 = clarifier.create_session(
        original_query="找绿色的东西",
        initial_clarifications=[
            ClarificationIntent(
                issue_type="missing_direction",
                description="缺少方向",
                confidence=0.8,
                priority=3,
                candidates=["北", "南", "东", "西"],
                severity="high"
            )
        ]
    )

    user_response2 = "北边"
    print(f"用户响应: {user_response2}")

    analysis = clarifier.analyze_user_response(session2.session_id, user_response2)
    print(f"解析结果:")
    print(f"  已解决: {analysis.resolved_issues}")
    print(f"  新实体: {analysis.new_entities}")
    print(f"  置信度: {analysis.confidence:.2f}")
    print(f"  需要澄清: {analysis.ambiguity_detected}")

    # 测试用例4：建议回答生成
    print("\n测试用例4: 建议回答生成")

    session3 = clarifier.create_session(
        original_query="测试",
        initial_clarifications=[
            ClarificationIntent(
                issue_type="missing_direction",
                description="缺少方向",
                confidence=0.8,
                priority=3,
                candidates=["北", "南", "东", "西"],
                severity="high"
            )
        ]
    )

    if session3.current_question:
        suggested = clarifier.generate_suggested_response(session3.session_id, num_variants=3)
        print(f"建议回答: {suggested}")

    # 测试用例5：统计信息
    print("\n测试用例5: 统计信息")

    stats = clarifier.get_stats()
    print(f"总澄清次数: {stats['total_clarifications']}")
    print(f"成功: {stats['total_confirmed']}")
    print(f"失败: {stats['total_failed']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"活跃会话: {stats['active_sessions']}")

    # 清理会话
    for session_id in list(clarifier.sessions.keys()):
        clarifier.delete_session(session_id)


def test_multi_scenario_clarifier():
    """测试多场景澄清器"""
    print("\n" + "=" * 60)
    print("测试多场景澄清")
    print("=" * 60)

    clarifier = InteractiveClarifier(mock_mode=True, language="zh")

    scenarios = [
        {
            "name": "场景1: 缺失方向",
            "query": "找红色的汽车",
            "intents": [
                ClarificationIntent(
                    issue_type="missing_direction",
                    description="缺少方向",
                    confidence=0.9,
                    priority=3,
                    candidates=["北", "南", "东", "西"],
                    severity="high"
                )
            ],
            "user_responses": ["北侧"]
        },
        {
            "name": "场景2: 模糊对象",
            "query": "找北边的东西",
            "intents": [
                ClarificationIntent(
                    issue_type="ambiguous_object",
                    description="对象模糊",
                    confidence=0.8,
                    priority=3,
                    candidates=["大楼", "柱子", "汽车", "树"],
                    severity="high"
                )
            ],
            "user_responses": ["柱子"]
        },
        {
            "name": "场景3: 多重澄清",
            "query": "找东西",
            "intents": [
                ClarificationIntent(
                    issue_type="missing_direction",
                    description="缺少方向",
                    confidence=0.9,
                    priority=3,
                    candidates=["北", "南", "东", "西"],
                    severity="high"
                ),
                ClarificationIntent(
                    issue_type="missing_color",
                    description="缺少颜色",
                    confidence=0.8,
                    priority=2,
                    candidates=["红", "绿", "蓝"],
                    severity="medium"
                )
            ],
            "user_responses": ["北侧", "红色"]
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"查询: {scenario['query']}")

        # 创建会话
        session = clarifier.create_session(
            original_query=scenario['query'],
            initial_clarifications=scenario['intents']
        )

        print(f"会话ID: {session.session_id}")
        if session.current_question:
            print(f"问题: {session.current_question.question_text}")

        # 模拟多轮交互
        for i, response in enumerate(scenario['user_responses'], 1):
            print(f"\n用户响应 {i}: {response}")
            completed, updated_data, followup_questions = clarifier.process_clarification_response(
                session.session_id, response
            )
            print(f"完成: {completed}")
            if completed:
                print(f"更新数据: {updated_data}")
                break
            elif followup_questions:
                print(f"后续问题: {followup_questions[0].question_text}")

        clarifier.delete_session(session.session_id)

    # 最终统计
    print(f"\n{'='*60}")
    print("最终统计")
    print(f"{'='*60}")

    stats = clarifier.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    test_interactive_clarifier()
    test_multi_scenario_clarifier()
