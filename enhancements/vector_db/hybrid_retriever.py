"""
混合检索协调器 - Text2Loc增强版

结合向量检索和模板匹配，提供更准确的检索结果
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    metadata: Dict[str, Any]  # 元数据
    score: float  # 综合得分
    vector_score: float  # 向量检索得分
    template_score: float  # 模板匹配得分
    rank: int  # 排名


class HybridRetriever:
    """混合检索协调器"""

    def __init__(self,
                 vector_weight: float = 0.65,
                 template_weight: float = 0.35,
                 enable_reranking: bool = True,
                 rerank_method: str = "weighted"):
        """
        初始化混合检索协调器

        Args:
            vector_weight: 向量检索权重 (从0.6提升到0.65，更侧重语义理解)
            template_weight: 模板匹配权重 (从0.4调整到0.35)
            enable_reranking: 是否启用重排序
            rerank_method: 重排序方法（weighted, combined）
        """
        self.vector_weight = vector_weight
        self.template_weight = template_weight
        self.enable_reranking = enable_reranking
        self.rerank_method = rerank_method

        # 模板匹配规则库
        self.template_rules = self._init_template_rules()

        # 统计信息
        self.total_queries = 0
        self.total_time = 0.0

        logger.info(f"混合检索协调器初始化完成: vector_weight={vector_weight}, template_weight={template_weight}")

    def _init_template_rules(self) -> Dict[str, List[str]]:
        """初始化模板匹配规则"""
        return {
            # 方向规则
            "direction_north": ["north", "北", "前方", "forward", "front"],
            "direction_south": ["south", "南", "后方", "backward", "back"],
            "direction_east": ["east", "东", "右侧", "right"],
            "direction_west": ["west", "西", "左侧", "left"],
            "direction_on_top": ["on top", "above", "over", "上", "上方", "顶部"],

            # 颜色规则
            "color_red": ["red", "红色", "红"],
            "color_green": ["green", "绿色", "绿"],
            "color_blue": ["blue", "蓝色", "蓝"],
            "color_gray": ["gray", "grey", "灰色", "灰"],
            "color_black": ["black", "黑色", "黑"],
            "color_white": ["white", "白色", "白"],

            # 对象规则
            "object_building": ["building", "大楼", "建筑", "建筑物", "楼房"],
            "object_pole": ["pole", "柱子", "灯柱", "电线杆", "杆子"],
            "object_parking": ["parking", "停车场", "停车位", "车位"],
            "object_sign": ["sign", "标志", "交通标志", "路标", "指示牌"],
            "object_light": ["light", "灯", "路灯", "照明"],
            "object_car": ["car", "汽车", "车辆", "小车"],
            "object_tree": ["tree", "树", "树木", "大树"],

            # 空间关系规则
            "relation_near": ["near", "beside", "next to", "附近", "旁边", "邻近"],
            "relation_between": ["between", "之间", "中间", "当中"],
            "relation_in_front": ["in front of", "前面", "前方", "正前方"],
            "relation_behind": ["behind", "后面", "后方", "背后"],
        }

    def _template_match_score(self, query_text: str, metadata: Dict[str, Any]) -> float:
        """
        计算模板匹配得分 - 增强版
    
        Args:
            query_text: 查询文本
            metadata: 候选元数据
    
        Returns:
            模板匹配得分 (0.0-1.0)
        """
        if not metadata:
            return 0.0
    
        query_lower = query_text.lower()
        scores = []
        matched_categories = 0  # 记录匹配的类别数
    
        # 检查方向匹配 - 提高权重
        if "direction" in metadata:
            direction = metadata["direction"]
            direction_lower = direction.lower()
    
            for rule_key, keywords in self.template_rules.items():
                if rule_key.startswith("direction_"):
                    for keyword in keywords:
                        # 使用更精确的匹配
                        if (keyword in query_lower and direction_lower in keyword) or \
                           (direction_lower in rule_key and keyword in query_text):
                            scores.append(1.0)
                            matched_categories += 1
                            break
                    if scores and scores[-1] == 1.0:
                        break
    
        # 检查颜色匹配 - 提高权重
        if "color" in metadata:
            color = metadata["color"]
            color_lower = color.lower()
    
            for rule_key, keywords in self.template_rules.items():
                if rule_key.startswith("color_"):
                    for keyword in keywords:
                        if (keyword in query_lower and color_lower in keyword) or \
                           (color_lower in rule_key and keyword in query_text):
                            scores.append(1.0)
                            matched_categories += 1
                            break
                    if scores and len(scores) > matched_categories - 1:
                        break
    
        # 检查对象匹配 - 核心匹配
        if "object" in metadata or "label" in metadata:
            obj = metadata.get("object") or metadata.get("label", "")
            obj_lower = obj.lower()
    
            for rule_key, keywords in self.template_rules.items():
                if rule_key.startswith("object_"):
                    for keyword in keywords:
                        if (keyword in query_lower and obj_lower in keyword) or \
                           (obj_lower in rule_key and keyword in query_text):
                            scores.append(1.0)
                            matched_categories += 1
                            break
                    if scores and len(scores) > matched_categories - 1:
                        break
    
        # 检查空间关系匹配
        if "relation" in metadata:
            relation = metadata["relation"]
            relation_lower = relation.lower()
    
            for rule_key, keywords in self.template_rules.items():
                if rule_key.startswith("relation_"):
                    for keyword in keywords:
                        if keyword in query_lower and relation_lower in keyword:
                            scores.append(0.85)  # 空间关系匹配权重略低
                            matched_categories += 1
                            break
                    if scores and len(scores) > matched_categories - 1:
                        break
    
        # 智能计算平均得分
        if scores:
            base_score = float(np.mean(scores))
                
            # 根据匹配类别数调整
            if matched_categories >= 3:
                bonus = 0.10  # 三个以上类别匹配，给予奖励
            elif matched_categories == 2:
                bonus = 0.05  # 两个类别匹配
            else:
                bonus = 0.0
                
            return min(1.0, base_score + bonus)
        else:
            # 部分匹配：检查是否有任何关键词匹配
            all_keywords = []
            for keywords in self.template_rules.values():
                all_keywords.extend(keywords)
    
            matched_keywords = sum(1 for keyword in set(all_keywords) if keyword in query_lower or keyword in query_text)
    
            if matched_keywords > 0:
                # 部分匹配得分，根据匹配数量调整
                return min(0.6, matched_keywords * 0.12)  # 提高部分匹配得分
            else:
                return 0.05  # 极低匹配

    def _calculate_template_scores(self, query_text: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        批量计算模板匹配得分

        Args:
            query_text: 查询文本
            candidates: 候选列表

        Returns:
            模板匹配得分列表
        """
        scores = []
        for candidate in candidates:
            score = self._template_match_score(query_text, candidate)
            scores.append(score)
        return scores

    def retrieve(self,
                 query_text: str,
                 query_embedding: np.ndarray,
                 vector_results: List[Tuple[Dict[str, Any], float]],
                 template_results: Optional[List[Dict[str, Any]]] = None,
                 top_k: int = 5) -> List[RetrievalResult]:
        """
        执行混合检索

        Args:
            query_text: 查询文本
            query_embedding: 查询向量
            vector_results: 向量检索结果 [(元数据, 相似度), ...]
            template_results: 模板匹配结果（可选）
            top_k: 返回前k个结果

        Returns:
            混合检索结果列表
        """
        start_time = time.time()

        try:
            logger.info(f"执行混合检索: '{query_text[:30]}...'")

            # 如果没有向量结果，返回空
            if not vector_results:
                return []

            # 提取向量检索结果
            vector_candidates = [r[0] for r in vector_results]
            vector_scores = [r[1] for r in vector_results]

            # 计算模板匹配得分
            template_scores = self._calculate_template_scores(query_text, vector_candidates)

            # 计算综合得分
            combined_results = []
            for i, (candidate, vec_score, temp_score) in enumerate(zip(vector_candidates, vector_scores, template_scores)):
                # 归一化得分
                norm_vec_score = vec_score  # 已经是0-1范围
                norm_temp_score = temp_score  # 已经是0-1范围

                # 加权综合得分
                combined_score = (
                    self.vector_weight * norm_vec_score +
                    self.template_weight * norm_temp_score
                )

                combined_results.append({
                    "metadata": candidate,
                    "combined_score": combined_score,
                    "vector_score": norm_vec_score,
                    "template_score": norm_temp_score
                })

            # 按综合得分排序
            combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

            # 重排序（如果启用）
            if self.enable_reranking:
                combined_results = self._rerank(combined_results, query_text)

            # 截取前k个结果
            top_results = combined_results[:top_k]

            # 转换为RetrievalResult
            retrieval_results = []
            for rank, result in enumerate(top_results, 1):
                retrieval_result = RetrievalResult(
                    metadata=result["metadata"],
                    score=result["combined_score"],
                    vector_score=result["vector_score"],
                    template_score=result["template_score"],
                    rank=rank
                )
                retrieval_results.append(retrieval_result)

            elapsed_time = time.time() - start_time
            self.total_queries += 1
            self.total_time += elapsed_time

            logger.info(f"混合检索完成: {len(retrieval_results)}个结果，耗时{elapsed_time:.3f}秒")

            return retrieval_results

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []

    def _rerank(self, candidates: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        """
        重排序候选结果

        Args:
            candidates: 候选结果列表
            query_text: 查询文本

        Returns:
            重排序后的候选结果列表
        """
        if self.rerank_method == "weighted":
            # 基于查询文本长度调整权重
            query_length = len(query_text)
            if query_length > 30:  # 长查询，提高向量权重
                adjusted_vector_weight = min(0.8, self.vector_weight + 0.2)
                adjusted_template_weight = 1.0 - adjusted_vector_weight
            else:  # 短查询，提高模板权重
                adjusted_template_weight = min(0.7, self.template_weight + 0.2)
                adjusted_vector_weight = 1.0 - adjusted_template_weight

            # 重新计算得分
            for result in candidates:
                result["combined_score"] = (
                    adjusted_vector_weight * result["vector_score"] +
                    adjusted_template_weight * result["template_score"]
                )

            # 重新排序
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        elif self.rerank_method == "combined":
            # 结合多种因素重排序
            for result in candidates:
                # 基础得分
                base_score = result["combined_score"]

                # 奖励高置信度结果
                if result["vector_score"] > 0.9 and result["template_score"] > 0.8:
                    bonus = 0.1
                elif result["vector_score"] > 0.8:
                    bonus = 0.05
                else:
                    bonus = 0.0

                result["combined_score"] = min(1.0, base_score + bonus)

            # 重新排序
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        return candidates

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self.total_time / self.total_queries if self.total_queries > 0 else 0

        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_time": avg_time,
            "config": {
                "vector_weight": self.vector_weight,
                "template_weight": self.template_weight,
                "enable_reranking": self.enable_reranking,
                "rerank_method": self.rerank_method
            }
        }

    def add_template_rule(self, rule_key: str, keywords: List[str]):
        """
        添加模板匹配规则

        Args:
            rule_key: 规则键
            keywords: 关键词列表
        """
        self.template_rules[rule_key] = keywords
        logger.info(f"添加模板规则: {rule_key}")

    def update_template_rules(self, new_rules: Dict[str, List[str]]):
        """
        更新模板规则库

        Args:
            new_rules: 新规则字典
        """
        self.template_rules.update(new_rules)
        logger.info(f"更新模板规则库，新增{len(new_rules)}条规则")
