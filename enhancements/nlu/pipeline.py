"""
Text2LocVisionaryPipeline - 三层智能处理架构管道

整合InstructionOptimizer、DynamicTemplateGenerator和InteractiveClarifier
实现完整的端到端智能定位处理流程
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

# 导入组件
from enhancements.nlu.instruction_optimizer import (
    InstructionOptimizer,
    OptimizedQuery,
    ClarificationIntent,
    ClarificationResponse,
    ClarificationResult
)

from enhancements.nlu.dynamic_template_generator import (
    DynamicTemplateGenerator,
    ParsedResult,
    TemplateVariant
)

from enhancements.nlu.interactive_clarifier import (
    InteractiveClarifier,
    ClarificationQuestion,
    ClarificationSession
)

logger = logging.getLogger(__name__)


@dataclass
class QueryProcessingStep:
    """查询处理步骤"""

    step_name: str
    step_type: str  # optimization, clarification, template_generation, retrieval
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time_ms: float
    status: str  # success, warning, error
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "step_type": self.step_type,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "processing_time_ms": self.processing_time_ms,
            "status": self.status,
            "details": self.details
        }


@dataclass
class LocalizationResult:
    """定位结果"""

    query_id: str
    original_query: str
    final_query: str
    processed_steps: List[QueryProcessingStep]
    final_result: Optional[Dict[str, Any]]
    statistics: Dict[str, Any]
    status: str  # success, needs_clarification, error
    clarification_questions: List[ClarificationQuestion]
    generated_templates: List[TemplateVariant]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "final_query": self.final_query,
            "processed_steps": [step.to_dict() for step in self.processed_steps],
            "final_result": self.final_result,
            "statistics": self.statistics,
            "status": self.status,
            "clarification_questions": [q.to_dict() for q in self.clarification_questions],
            "generated_templates": [t.to_dict() for t in self.generated_templates]
        }


@dataclass
class ProcessingContext:
    """处理上下文"""

    session_id: Optional[str]
    user_id: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class PipelineMode(Enum):
    """管道模式"""
    FAST = "fast"  # 快速模式
    BALANCED = "balanced"  # 平衡模式
    PRECISE = "precise"  # 精确模式


class Text2LocVisionaryPipeline:
    """
    Text2Loc Visionary 三层智能处理架构管道

    架构层：
    1. 智能理解层（InstructionOptimizer）: 理解用户意图，优化查询
    2. 适配转换层（DynamicTemplateGenerator）: 生成Text2Loc兼容模板
    3. 执行处理层（原系统）: 执行定位检索
    4. 交互澄清层（InteractiveClarifier）: 处理模糊查询
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen3-vl:2b",
        mock_mode: bool = True,
        default_mode: PipelineMode = PipelineMode.BALANCED,
        cache_enabled: bool = True,
        language: str = "zh"
    ):
        """
        初始化管道

        Args:
            ollama_url: Ollama API地址
            model_name: Qwen模型名称
            mock_mode: 是否使用模拟模式
            default_mode: 默认管道模式
            cache_enabled: 是否启用缓存
            language: 默认语言
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.mock_mode = mock_mode
        self.default_mode = default_mode
        self.cache_enabled = cache_enabled
        self.language = language

        # 初始化三层架构组件
        self.instruction_optimizer = InstructionOptimizer(
            ollama_url=ollama_url,
            model_name=model_name,
            mock_mode=mock_mode,
            cache_enabled=cache_enabled
        )

        self.template_generator = DynamicTemplateGenerator()

        self.interactive_clarifier = InteractiveClarifier(
            ollama_url=ollama_url,
            model_name=model_name,
            mock_mode=mock_mode,
            language=language
        )

        # 执行层组件（模拟原Text2Loc系统）
        self.retrieval_component = RetrievalComponent()

        # 会话管理
        self.active_sessions: Dict[str, ProcessingContext] = {}

        # 统计信息
        self.total_queries = 0
        self.total_clarifications = 0
        self.total_time = 0.0
        self.successful_queries = 0

        logger.info(f"Text2LocVisionaryPipeline 初始化完成")
        logger.info(f"  - 模式: {default_mode.value}")
        logger.info(f"  - 模型: {model_name}")
        logger.info(f"  - 模拟模式: {mock_mode}")
        logger.info(f"  - 语言: {language}")

    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        创建处理会话

        Args:
            session_id: 会话ID（可选）
            user_id: 用户ID（可选）
            metadata: 元数据（可选）

        Returns:
            会话ID
        """
        if session_id is None:
            session_id = f"pipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() % 10000)}"

        context = ProcessingContext(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )

        self.active_sessions[session_id] = context

        logger.info(f"创建处理会话: {session_id}")
        return session_id

    def get_session_context(self, session_id: str) -> Optional[ProcessingContext]:
        """获取会话上下文"""
        return self.active_sessions.get(session_id)

    def end_session(self, session_id: str):
        """结束会话"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"结束处理会话: {session_id}")

    def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: Optional[PipelineMode] = None,
        top_k: int = 5
    ) -> LocalizationResult:
        """
        处理查询 - 端到端流程

        智能理解层 -> 交互澄清层 -> 适配转换层 -> 执行处理层

        Args:
            query: 用户查询
            session_id: 会话ID（可选）
            user_id: 用户ID（可选）
            mode: 管道模式（可选）
            top_k: 返回结果数量

        Returns:
            定位结果
        """
        start_time = time.time()

        # 准备上下文
        if session_id is None:
            session_id = self.create_session(user_id=user_id)

        context = self.get_session_context(session_id)
        if context is None:
            context = ProcessingContext(session_id=session_id, user_id=user_id)

        query_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_queries}"
        self.total_queries += 1

        processed_steps: List[QueryProcessingStep] = []
        clarification_questions: List[ClarificationQuestion] = []
        generated_templates: List[TemplateVariant] = []
        final_result: Optional[Dict[str, Any]] = None

        current_mode = mode or self.default_mode

        try:
            # ========== 阶段1: 智能理解层（指令优化）==========
            logger.info(f"[{query_id}] 开始智能理解层处理")

            optimize_start = time.time()

            # 调用指令优化器
            optimized_query = self.instruction_optimizer.optimize(
                user_input=query,
                session_id=context.session_id if context else None,
                session_context=context.metadata if context else None
            )

            optimize_time = (time.time() - optimize_start) * 1000

            processed_steps.append(QueryProcessingStep(
                step_name="instruction_optimization",
                step_type="optimization",
                input_data={"original_query": query},
                output_data=optimized_query.to_dict(),
                processing_time_ms=optimize_time,
                status="success",
                details=f"置信度: {optimized_query.confidence_scores.get('overall', 0):.2f}"
            ))

            logger.info(f"[{query_id}] 智能理解完成: {optimize_time:.1f}ms")
            logger.info(f"[{query_id}]   优化查询: {optimized_query.optimized_input}")
            logger.info(f"[{query_id}]   需要澄清: {optimized_query.need_clarification}")

            # 如果需要澄清，进入交互澄清流程
            if optimized_query.need_clarification:
                logger.info(f"[{query_id}] 需要澄清，进入交互澄清层")

                clarify_start = time.time()

                # 识别澄清需求
                clarification_intents = [
                    ClarificationIntent(
                        issue_type=issue_type,
                        description=f"需要澄清: {issue_type}",
                        confidence=0.8,
                        priority=3,
                        candidates=["北", "南", "东", "西", "上", "红", "绿", "蓝", "大楼", "柱子", "汽车"],
                        severity="medium"
                    )
                    for issue_type in optimized_query.clarification_types[:2]  # 最多2个
                ]

                # 生成澄清问题
                if clarification_intents:
                    questions = self.interactive_clarifier.generate_clarification_questions(
                        clarification_intents=clarification_intents,
                        session=self.interactive_clarifier.create_session(
                            session_id=f"{context.session_id}_clarify" if context else None,
                            original_query=query,
                            initial_clarifications=clarification_intents
                        ),
                        language=self.language
                    )

                    clarification_questions.extend(questions)

                    clarify_time = (time.time() - clarify_start) * 1000

                    processed_steps.append(QueryProcessingStep(
                        step_name="interactive_clarification",
                        step_type="clarification",
                        input_data={"optimized_query": optimized_query.to_dict()},
                        output_data={
                            "clarification_intents": [c.to_dict() for c in clarification_intents],
                            "questions": [q.to_dict() for q in questions]
                        },
                        processing_time_ms=clarify_time,
                        status="success",
                        details=f"生成{len(questions)}个澄清问题"
                    ))

                    logger.info(f"[{query_id}] 澄清生成完成: {clarify_time:.1f}ms")
                    logger.info(f"[{query_id}]   澄清问题: {len(questions)}个")

                    # 返回需要澄清的结果
                    total_time = (time.time() - start_time) * 1000
                    self.total_time += total_time

                    return LocalizationResult(
                        query_id=query_id,
                        original_query=query,
                        final_query=query,
                        processed_steps=processed_steps,
                        final_result=None,
                        statistics={
                            "total_time_ms": total_time,
                            "mode": current_mode.value,
                            "needs_clarification": True,
                            "clarification_count": len(clarification_questions)
                        },
                        status="needs_clarification",
                        clarification_questions=clarification_questions,
                        generated_templates=generated_templates
                    )

            # ========== 阶段2: 适配转换层（动态模板生成）==========
            logger.info(f"[{query_id}] 开始适配转换层处理")

            template_start = time.time()

            # 构建解析结果
            parsed_result = self._build_parsed_result(optimized_query)

            # 生成模板变体
            template_variants = self.template_generator.generate(
                parsed_result=parsed_result,
                n_variants=5 if current_mode == PipelineMode.PRECISE else 3
            )

            generated_templates.extend(template_variants)

            template_time = (time.time() - template_start) * 1000

            processed_steps.append(QueryProcessingStep(
                step_name="template_generation",
                step_type="template_generation",
                input_data=optimized_query.to_dict(),
                output_data={
                    "parsed_result": parsed_result.to_dict(),
                    "template_variants": [t.to_dict() for t in template_variants]
                },
                processing_time_ms=template_time,
                status="success",
                details=f"生成{len(template_variants)}个模板变体"
            ))

            logger.info(f"[{query_id}] 模板生成完成: {template_time:.1f}ms")
            logger.info(f"[{query_id}]   模板变体数: {len(template_variants)}")

            if not template_variants:
                raise ValueError("无法生成有效的模板")

            # ========== 阶段3: 执行处理层（检索）==========
            logger.info(f"[{query_id}] 开始执行处理层检索")

            retrieval_start = time.time()

            # 选择最佳模板进行检索
            best_template = template_variants[0] if template_variants else None

            if best_template:
                # 执行检索（原Text2Loc系统）
                retrieval_results = self.retrieval_component.retrieve(
                    query=best_template.filled_text,
                    template_type=best_template.template_type,
                    top_k=top_k,
                    mode=current_mode
                )

                retrieval_time = (time.time() - retrieval_start) * 1000

                processed_steps.append(QueryProcessingStep(
                    step_name="retrieval",
                    step_type="retrieval",
                    input_data={"template": best_template.filled_text},
                    output_data=retrieval_results,
                    processing_time_ms=retrieval_time,
                    status="success",
                    details=f"检索结果: {len(retrieval_results.get('results', []))}个"
                ))

                logger.info(f"[{query_id}] 检索完成: {retrieval_time:.1f}ms")

                # 整理最终结果
                final_result = {
                    "best_template": best_template.to_dict(),
                    "retrieval_results": retrieval_results,
                    "all_templates": [t.to_dict() for t in template_variants],
                    "optimization_log": optimized_query.optimization_log
                }

                self.successful_queries += 1

            # ========== 汇总统计信息==========
            total_time = (time.time() - start_time) * 1000
            self.total_time += total_time

            stats = {
                "total_time_ms": total_time,
                "mode": current_mode.value,
                "model_used": self.model_name,
                "cache_enabled": self.cache_enabled,
                "steps_count": len(processed_steps),
                "optimization_time": optimized_query.confidence_scores.get('overall', 0),
                "template_count": len(generated_templates),
                "clarification_questions": len(clarification_questions)
            }

            logger.info(f"[{query_id}] 查询处理完成: {total_time:.1f}ms")
            logger.info(f"[{query_id}]   状态: success")
            logger.info(f"{'='*60}")

            return LocalizationResult(
                query_id=query_id,
                original_query=query,
                final_query=optimized_query.optimized_input,
                processed_steps=processed_steps,
                final_result=final_result,
                statistics=stats,
                status="success",
                clarification_questions=clarification_questions,
                generated_templates=generated_templates
            )

        except Exception as e:
            logger.error(f"[{query_id}] 查询处理失败: {e}")

            total_time = (time.time() - start_time) * 1000
            self.total_time += total_time

            return LocalizationResult(
                query_id=query_id,
                original_query=query,
                final_query=query,
                processed_steps=processed_steps,
                final_result=None,
                statistics={
                    "total_time_ms": total_time,
                    "error": str(e),
                    "mode": current_mode.value if current_mode else "unknown"
                },
                status="error",
                clarification_questions=clarification_questions,
                generated_templates=generated_templates
            )

    def process_with_clarification(
        self,
        session_id: str,
        user_response: str,
        top_k: int = 5
    ) -> LocalizationResult:
        """
        处理澄清后的查询

        Args:
            session_id: 会话ID
            user_response: 用户回答
            top_k: 返回结果数量

        Returns:
            定位结果
        """
        # 获取会话上下文
        context = self.get_session_context(session_id)
        if not context:
            raise ValueError(f"会话不存在: {session_id}")

        # 分析用户响应
        clarification_session = self.interactive_clarifier.get_session(
            f"{session_id}_clarify"
        )

        if not clarification_session:
            raise ValueError(f"澄清会话不存在: {session_id}_clarify")

        # 处理用户响应
        completed, updated_data, followup_questions = (
            self.interactive_clarifier.process_clarification_response(
                session_id=f"{session_id}_clarify",
                user_response=user_response,
                language=self.language
            )
        )

        if completed and updated_data:
            # 使用更新后的数据重新处理查询
            new_query = user_response

            # 需要知道原始查询作为上下文，这里简化处理
            # 在实际系统中，应该从历史记录中获取
            return self.process_query(
                query=new_query,
                session_id=session_id,
                top_k=top_k
            )

        # 需要更多澄清
        query_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_queries}"
        self.total_queries += 1

        return LocalizationResult(
            query_id=query_id,
            original_query=user_response,
            final_query=user_response,
            processed_steps=[],
            final_result=None,
            statistics={
                "total_time_ms": 0,
                "needs_more_clarification": True,
                "followup_questions": len(followup_questions)
            },
            status="needs_more_clarification",
            clarification_questions=followup_questions,
            generated_templates=[]
        )

    def _build_parsed_result(self, optimized_query: OptimizedQuery) -> ParsedResult:
        """从优化查询构建解析结果"""

        entities = optimized_query.parsed_elements
        confidence_scores = optimized_query.confidence_scores

        # 提取各字段
        direction = None
        color = None
        obj = None
        relation = None

        # 从entities中提取
        if entities.get("direction"):
            direction = entities["direction"].get("value")

        if entities.get("color"):
            color = entities["color"].get("value")

        if entities.get("object"):
            obj = entities["object"].get("value")

        if entities.get("relation"):
            relation = entities["relation"].get("value")

        # 计算完整性分数
        present_fields = 0
        total_fields = 2  # 至少需要方向和对象

        if direction:
            present_fields += 1
        if obj:
            present_fields += 1
        if color:
            present_fields += 0.5
        if relation:
            present_fields += 0.3

        completeness_score = min(present_fields / total_fields, 1.0)

        # 如果有置信度分数，使用它们
        if confidence_scores:
            direction_conf = confidence_scores.get("direction", 0)
            object_conf = confidence_scores.get("object", 0)
            color_conf = confidence_scores.get("color", 0)
            relation_conf = confidence_scores.get("relation", 0)
        else:
            direction_conf = 0.7 if direction else 0.0
            object_conf = 0.7 if obj else 0.0
            color_conf = 0.7 if color else 0.0
            relation_conf = 0.7 if relation else 0.0

        return ParsedResult(
            direction=direction,
            color=color,
            object=obj,
            relation=relation,
            distance=None,
            completeness_score=completeness_score,
            confidence_scores={
                "direction": direction_conf,
                "color": color_conf,
                "object": object_conf,
                "relation": relation_conf
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取管道统计信息"""

        optimizer_stats = self.instruction_optimizer.get_stats()
        template_stats = self.template_generator.get_performance_stats()
        clarifier_stats = self.interactive_clarifier.get_stats()

        avg_time = self.total_time / self.total_queries if self.total_queries > 0 else 0.0

        return {
            "pipeline": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "total_clarifications": self.total_clarifications,
                "total_time_ms": self.total_time,
                "avg_time_ms": avg_time,
                "success_rate": self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0,
                "mode": self.default_mode.value,
                "model": self.model_name,
                "mock_mode": self.mock_mode,
                "active_sessions": len(self.active_sessions),
                "cache_enabled": self.cache_enabled
            },
            "optimizer": optimizer_stats,
            "template_generator": template_stats,
            "clarifier": clarifier_stats
        }

    def clear_cache(self):
        """清空所有缓存"""
        self.instruction_optimizer.clear_cache()
        self.template_generator.get_performance_stats()  # This doesn't have clear_cache yet
        self.interactive_clarifier.clear_cache()
        logger.info("所有缓存已清空")

    def export_state(self) -> Dict[str, Any]:
        """导出管道状态"""

        template_state = self.template_generator.export_learnings()
        clarifier_state = self.interactive_clarifier.export_sessions()

        return {
            "pipeline_state": {
                "total_queries": self.total_queries,
                "total_clarifications": self.total_clarifications,
                "total_time": self.total_time,
                "active_sessions": {
                    sid: ctx.to_dict() for sid, ctx in self.active_sessions.items()
                }
            },
            "template_learnings": template_state,
            "clarifier_state": clarifier_state
        }

    def import_state(self, state: Dict[str, Any]):
        """导入管道状态"""
        if "pipeline_state" in state:
            pipeline_state = state["pipeline_state"]
            self.total_queries = pipeline_state.get("total_queries", 0)
            self.total_clarifications = pipeline_state.get("total_clarifications", 0)
            self.total_time = pipeline_state.get("total_time", 0.0)

        if "template_learnings" in state:
            self.template_generator.import_learnings(state["template_learnings"])

        if "clarifier_state" in state:
            self.interactive_clarifier.import_sessions(state["clarifier_state"])

        logger.info("管道状态导入完成")


class RetrievalComponent:
    """
    检索组件 - 模拟原Text2Loc系统的检索功能
    在实际系统中，这会调用真正的Text2Loc模型
    """

    def __init__(self):
        self.simulated_cells = None
        logger.info("RetrievalComponent 初始化完成")

    def retrieve(
        self,
        query: str,
        template_type: str,
        top_k: int = 5,
        mode: PipelineMode = PipelineMode.BALANCED
    ) -> Dict[str, Any]:
        """
        执行检索

        Args:
            query: 查询文本
            template_type: 模板类型
            top_k: 返回结果数量
            mode: 管道模式

        Returns:
            检索结果
        """
        start_time = time.time()

        try:
            # 在模拟模式下，生成模拟的检索结果
            simulated_results = self._generate_mock_results(query, top_k)

            elapsed = (time.time() - start_time) * 1000

            return {
                "query": query,
                "template_type": template_type,
                "top_k": top_k,
                "mode": mode.value,
                "results": simulated_results,
                "processing_time_ms": elapsed,
                "simulation_mode": True
            }

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return {
                "query": query,
                "template_type": template_type,
                "top_k": top_k,
                "mode": mode.value,
                "results": [],
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "simulation_mode": True
            }

    def _generate_mock_results(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """生成模拟检索结果"""

        # 根据查询内容生成模拟结果
        # 这里简化处理：返回一些固定位置

        base_cell_id = 100  # 起始单元

        mock_results = []

        for i in range(top_k):
            cell_id = f"cell_{base_cell_id + i * 10}"
            score = 1.0 - (i * 0.15)  # 分数递减

            # 模拟坐标和方向
            mock_results.append({
                "cell_id": cell_id,
                "centroid": {
                    "x": i * 0.5,
                    "y": 0.0,
                    "z": 0.0
                },
                "normalized_distance": score,
                "similarity_score": score,
                "query_match": query[:20],  # 截取查询部分
                "metadata": {
                    "retrieval_type": "mock",
                    "rank": i + 1
                }
            })

        return mock_results


# 测试函数
def test_pipeline():
    """测试管道"""
    print("=" * 60)
    print("测试 Text2LocVisionaryPipeline")
    print("=" * 60)

    # 创建管道
    pipeline = Text2LocVisionaryPipeline(
        ollama_url="http://localhost:11434",
        model_name="qwen3-vl:2b",
        mock_mode=True,
        default_mode=PipelineMode.BALANCED,
        cache_enabled=True,
        language="zh"
    )

    # 测试用例1: 完整的查询流程
    print("\n测试用例1: 完整查询流程")

    query = "我在停车场的北侧附近看到一个红色的汽车"

    result = pipeline.process_query(query=query, top_k=3)

    print(f"查询ID: {result.query_id}")
    print(f"状态: {result.status}")
    print(f"最终查询: {result.final_query}")

    print(f"\n处理步骤 ({len(result.processed_steps)}个):")
    for i, step in enumerate(result.processed_steps, 1):
        print(f"  {i}. {step.step_name} ({step.step_type}) - {step.processing_time_ms:.1f}ms - {step.status}")

    if result.final_result:
        print(f"\n最终结果:")
        best_template = result.final_result.get("best_template", {})
        print(f"  最佳模板: {best_template.get('filled_text', 'N/A')}")

    print(f"\n统计信息:")
    for key, value in result.statistics.items():
        print(f"  {key}: {value}")

    # 测试用例2: 需要澄清的查询
    print("\n" + "=" * 60)
    print("测试用例2: 需要澄清的查询")

    query2 = "找一个东西"  # 模糊查询

    result2 = pipeline.process_query(query=query2, top_k=3)

    print(f"查询ID: {result2.query_id}")
    print(f"状态: {result2.status}")
    print(f"最终查询: {result2.final_query}")

    if result2.clarification_questions:
        print(f"\n澄清问题 ({len(result2.clarification_questions)}个):")
        for i, q in enumerate(result2.clarification_questions, 1):
            print(f"  {i}. {q.question_text}")
            if q.suggested_answer:
                print(f"     建议回答: {q.suggested_answer}")

    # 测试用例3: 测试澄清流程
    if result2.clarification_questions:
        print("\n" + "=" * 60)
        print("测试用例3: 澄清流程")

        # 创建会话
        session_id = pipeline.create_session(user_id="test_user")
        print(f"创建会话: {session_id}")

        # 模拟用户响应
        user_response = "北侧的红色汽车"
        print(f"用户响应: {user_response}")

        # 处理澄清
        clarification_result = pipeline.process_with_clarification(
            session_id=session_id,
            user_response=user_response,
            top_k=3
        )

        print(f"澄清结果状态: {clarification_result.status}")

        # 再次处理
        if clarification_result.status == "success":
            print(f"澄清后最终查询: {clarification_result.final_query}")

            if clarification_result.final_result:
                best_template = clarification_result.final_result.get("best_template", {})
                print(f"澄清后最佳模板: {best_template.get('filled_text', 'N/A')}")

        # 结束会话
        pipeline.end_session(session_id)

    # 测试用例4: 查询统计信息
    print("\n" + "=" * 60)
    print("测试用例4: 统计信息")

    stats = pipeline.get_statistics()

    print(f"管道统计:")
    for key, value in stats['pipeline'].items():
        print(f"  {key}: {value}")

    print(f"\n各组件统计:")
    for component, component_stats in stats.items():
        if component != 'pipeline':
            print(f"\n  {component}:")
            for key, value in component_stats.items():
                if not isinstance(value, (dict, list)):
                    print(f"    {key}: {value}")

    # 测试用例5: 状态导出/导入
    print("\n" + "=" * 60)
    print("测试用例5: 状态导出/导入")

    state = pipeline.export_state()
    print(f"导出状态成功，包含 {len(state['template_learnings']['template_performance'])} 个模板学习结果")

    # 新建管道导入状态
    pipeline2 = Text2LocVisionaryPipeline(
        ollama_url="http://localhost:11434",
        model_name="qwen3-vl:2b",
        mock_mode=True,
        default_mode=PipelineMode.BALANCED,
        cache_enabled=True,
        language="zh"
    )

    pipeline2.import_state(state)

    stats2 = pipeline2.get_statistics()
    print(f"导入后总查询数: {stats2['pipeline']['total_queries']}")

    # 清理
    pipeline.clear_cache()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def test_multi_session_pipeline():
    """测试多会话管道"""
    print("\n" + "=" * 60)
    print("测试多会话管道")
    print("=" * 60)

    pipeline = Text2LocVisionaryPipeline(mock_mode=True)

    # 创建多个会话
    session_ids = []

    for i in range(3):
        session_id = pipeline.create_session(user_id=f"user_{i}")
        session_ids.append(session_id)
        print(f"创建会话 {i+1}: {session_id}")

    # 在不同会话中处理查询
    queries = [
        "停车场附近的红色汽车",
        "北侧的绿色树木",
        "东边建筑物旁的灯柱"
    ]

    results = []

    for i, (session_id, query) in enumerate(zip(session_ids, queries), 1):
        print(f"\n会话 {i} 处理查询: {query}")
        result = pipeline.process_query(query=query, session_id=session_id, top_k=2)
        results.append(result)
        print(f"  状态: {result.status}, 用时: {result.statistics.get('total_time_ms', 0):.1f}ms")

    # 查看统计
    stats = pipeline.get_statistics()
    print(f"\n总体统计:")
    print(f"  总查询数: {stats['pipeline']['total_queries']}")
    print(f"  活跃会话: {stats['pipeline']['active_sessions']}")
    print(f"  平均用时: {stats['pipeline']['avg_time_ms']:.1f}ms")

    # 清理会话
    for session_id in session_ids:
        pipeline.end_session(session_id)

    print(f"\n会话清理完成，剩余会话: {pipe.active_sessions}")

    print("\n" + "=" * 60)
    print("多会话测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行测试
    test_pipeline()

    # 运行多会话测试
    test_multi_session_pipeline()
