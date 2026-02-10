"""
Text2Loc 原始系统集成适配器
连接 Text2Loc Visionary API 与原始 Text2Loc 系统

优化版本：使用OptimizedCellRetrieval模型
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 添加原始 Text2Loc 到路径
TEXT2LOC_ORIGINAL_PATH = Path("d:/Text2Loc-main/Text2Loc-main")
if str(TEXT2LOC_ORIGINAL_PATH) not in sys.path:
    sys.path.insert(0, str(TEXT2LOC_ORIGINAL_PATH))

# 导入优化模型
try:
    from models.optimized_retrieval import get_optimized_retrieval, OptimizedRetrievalConfig
    OPTIMIZED_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"优化模型未找到: {e}")
    OPTIMIZED_MODEL_AVAILABLE = False

class Text2LocAdapter:
    """
    Text2Loc 原始系统适配器
    提供真实的位置定位功能
    """
    
    def __init__(self, data_path: str = None):
        """
        初始化适配器
        
        Args:
            data_path: KITTI360Pose 数据集路径
        """
        # 自动检测数据路径（优先使用带语义标签的新数据集）
        if data_path:
            self.data_path = data_path
        else:
            # 尝试多个可能的数据路径（优先使用k360_semantic）
            possible_paths = [
                "~/Desktop/Text2Loc-main/data/k360_semantic",  # 带语义标签的新数据集
                "./data/k360_semantic",
                "../data/k360_semantic",
                "~/Desktop/Text2Loc-main/data/k360_repaired",  # 修复后的数据集
                "./data/k360_repaired",
                "../data/k360_repaired",
                "~/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all",  # 原始数据集
                "d:/Text2Loc-main/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all",
                "./data/k360_30-10_scG_pd10_pc4_spY_all",
                "../data/k360_30-10_scG_pd10_pc4_spY_all",
            ]

            self.data_path = None
            for path in possible_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    self.data_path = expanded_path
                    logger.info(f"📁 找到数据路径: {self.data_path}")
                    break

            if not self.data_path:
                self.data_path = os.path.expanduser("~/Desktop/Text2Loc-main/data/k360_semantic")
                logger.warning(f"⚠️ 使用默认数据路径: {self.data_path}")
        self.cells = {}
        self.poses = {}
        self.directions = {}
        self.scene_names = []
        
        # 加载数据
        self._load_data()
        
        logger.info(f"✅ Text2LocAdapter 初始化完成")
        logger.info(f"   场景数量: {len(self.scene_names)}")
        logger.info(f"   Cell 总数: {sum(len(cells) for cells in self.cells.values())}")
    
    def _load_data(self):
        """加载 KITTI360Pose 数据集"""
        try:
            cells_path = Path(self.data_path) / "cells"
            poses_path = Path(self.data_path) / "poses"
            direction_path = Path(self.data_path) / "direction"
            
            if not cells_path.exists():
                logger.warning(f"⚠️ 数据路径不存在: {self.data_path}")
                logger.warning("   将使用模拟模式")
                return
            
            # 加载所有场景的 cells 和 poses
            for pkl_file in cells_path.glob("*.pkl"):
                scene_name = pkl_file.stem
                self.scene_names.append(scene_name)
                
                # 加载 cells
                with open(pkl_file, 'rb') as f:
                    self.cells[scene_name] = pickle.load(f)
                
                # 加载 poses
                pose_file = poses_path / f"{scene_name}.pkl"
                if pose_file.exists():
                    with open(pose_file, 'rb') as f:
                        self.poses[scene_name] = pickle.load(f)
                
                # 加载方向信息
                dir_file = direction_path / f"{scene_name}.json"
                if dir_file.exists():
                    import json
                    with open(dir_file, 'r') as f:
                        self.directions[scene_name] = json.load(f)
                
                logger.info(f"   已加载场景: {scene_name}")
                
        except Exception as e:
            logger.error(f"❌ 加载数据失败: {e}")
            logger.warning("   将使用模拟模式")
    
    def find_location(self, 
                      query: str, 
                      direction: str = None,
                      color: str = None,
                      obj: str = None,
                      top_k: int = 3,
                      use_optimized: bool = True) -> List[Dict[str, Any]]:
        """
        根据查询找到位置
        
        Args:
            query: 自然语言查询
            direction: 方向（north, south, east, west 等）
            color: 颜色
            obj: 对象名称
            top_k: 返回结果数量
            use_optimized: 是否使用优化模型
            
        Returns:
            候选位置列表，每个包含坐标和置信度
        """
        # 优先使用优化模型（如果启用且可用）
        if use_optimized and OPTIMIZED_MODEL_AVAILABLE:
            try:
                retrieval = get_optimized_retrieval()
                results = retrieval.retrieve(query, direction, color, obj, top_k)
                if results:
                    logger.debug(f"优化模型返回 {len(results)} 个结果")
                    return results
            except Exception as e:
                logger.warning(f"优化模型调用失败，回退到传统方法: {e}")
        
        if not self.cells:
            # 模拟模式
            return self._mock_find_location(query, direction, color, obj, top_k)
        
        # 真实模式：基于解析的参数搜索匹配的 cells
        candidates = []
        
        # 词汇映射：将常见查询词映射到数据集中的物体标签
        # 更新为与启发式推断的标签一致
        object_mapping = {
            # 交通相关
            'pedestrian crossing': 'road',
            'crosswalk': 'road',
            'bus stop': 'road',
            'traffic light': 'traffic light',
            'traffic sign': 'traffic sign',
            'road sign': 'traffic sign',
            'street lamp': 'lamp',
            'stop sign': 'stop',
            # 车辆相关
            'car': 'road',  # 车辆通常在road上
            'vehicle': 'road',
            'truck': 'road',
            'bus': 'road',
            'bicycle': 'road',
            'bike': 'road',
            'motorcycle': 'road',
            'parked car': 'road',
            # 建筑相关
            'building': 'building',
            'storefront': 'building',
            'house': 'building',
            'garage': 'garage',
            'wall': 'wall',
            'fence': 'fence',
            # 自然相关
            'tree': 'vegetation',
            'trees': 'vegetation',
            'green tree': 'vegetation',
            'park': 'vegetation',
            'grass': 'vegetation',
            'terrain': 'terrain',
            # 基础设施
            'sidewalk': 'sidewalk',
            'bridge': 'bridge',
            'tunnel': 'tunnel',
            'intersection': 'road',
            'roundabout': 'road',
            'parking lot': 'parking',
            'parking': 'parking',
            'road': 'road',
            'street': 'road',
            'gas station': 'building',
            'construction site': 'building',
            'school zone': 'building',
            # 其他
            'pole': 'pole',
            'trash bin': 'trash bin',
            'box': 'box',
            'vending machine': 'vending machine',
        }
        
        # 场景名称列表（用于从查询中提取场景）
        SCENE_NAMES = [
            '2013_05_28_drive_0000_sync',
            '2013_05_28_drive_0002_sync',
            '2013_05_28_drive_0003_sync',
            '2013_05_28_drive_0004_sync',
            '2013_05_28_drive_0005_sync',
            '2013_05_28_drive_0006_sync',
            '2013_05_28_drive_0007_sync',
            '2013_05_28_drive_0009_sync',
            '2013_05_28_drive_0010_sync',
        ]
        
        # 从查询中提取场景名称
        query_scene = None
        query_lower = query.lower()
        for scene in SCENE_NAMES:
            if scene.lower() in query_lower:
                query_scene = scene
                break
        
        # 如果obj无法识别，尝试从查询文本中提取关键词
        mapped_obj = obj
        if (not obj or obj == 'none') and query:
            query_lower = query.lower()
            for key, value in object_mapping.items():
                if key in query_lower:
                    mapped_obj = value
                    break
        
        # 使用映射后的obj进行搜索
        search_obj = mapped_obj if mapped_obj and mapped_obj != 'none' else obj
        
        for scene_name, cells in self.cells.items():
            for cell in cells:
                # 场景匹配权重：如果是查询中指定的场景，给予更高权重
                scene_bonus = 1.0  # 默认无场景加成
                if query_scene:
                    if scene_name == query_scene or scene_name in query_scene or query_scene in scene_name:
                        scene_bonus = 1.5  # 目标场景增加50%分数
                    else:
                        scene_bonus = 0.6  # 非目标场景降低40%分数
                
                score, best_obj = self._calculate_match_score_with_object(cell, direction, color, search_obj)
                
                # 应用场景权重
                score = score * scene_bonus
                
                if score > 0.10:  # 调整阈值
                    # 获取最佳匹配object的精确坐标（而不是cell中心）
                    x, y = self._get_best_object_center(cell, best_obj, direction, color, search_obj)
                    
                    # 获取真实的Cell ID
                    if isinstance(cell, dict):
                        cell_id = cell.get('id', f"cell_{len(candidates):03d}")
                    else:
                        cell_id = getattr(cell, 'id', f"cell_{len(candidates):03d}")
                    
                    candidates.append({
                        "cell_id": cell_id,
                        "scene": scene_name,
                        "score": score,
                        "x": x,
                        "y": y,
                        "confidence": score,
                        "description": self._generate_description(cell, direction, color, obj),
                        "reference_objects": self._get_reference_objects(cell, obj),
                        "scene_bonus": scene_bonus  # 记录是否有场景加成
                    })
        
        # 按分数排序
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # 如果没有找到任何候选，返回默认结果（避免空结果）
        if not candidates:
            logger.warning(f"未找到匹配结果，返回默认候选: query={query}, obj={search_obj}")
            # 返回得分最高的几个cell作为默认结果
            default_candidates = []
            count = 0
            for scene_name, cells in self.cells.items():
                for cell in cells:
                    if count >= top_k:
                        break
                    x, y = self._get_cell_center(cell)
                    if isinstance(cell, dict):
                        cell_id = cell.get('id', f"cell_{count:03d}")
                    else:
                        cell_id = getattr(cell, 'id', f"cell_{count:03d}")
                    
                    default_candidates.append({
                        "cell_id": cell_id,
                        "scene": scene_name,
                        "score": 0.1,
                        "x": x,
                        "y": y,
                        "confidence": 0.1,
                        "description": f"默认结果: {query}",
                        "reference_objects": []
                    })
                    count += 1
                if count >= top_k:
                    break
            return default_candidates
        
        return candidates[:top_k]
    
    def _calculate_match_score(self, cell, direction: str, color: str, obj: str) -> float:
        """计算 cell 与查询的匹配分数 - 真实数据版"""
        score = 0.0
        
        # 支持字典和对象两种格式
        cell_objects = cell.get('objects', []) if isinstance(cell, dict) else (getattr(cell, 'objects', []) if hasattr(cell, 'objects') else [])
        
        if not cell_objects:
            return 0.05  # 空cell也给基础分
        
        # 增强对象匹配 - 支持多关键词和模糊匹配
        if obj and obj != 'none':
            obj_lower = obj.lower()
            best_obj_score = 0.0
            
            for o in cell_objects:
                # 支持字典和对象格式，处理数组类型
                if isinstance(o, dict):
                    label_raw = o.get('label', '')
                    class_name_raw = o.get('class_name', '')
                    
                    label = str(label_raw).lower() if label_raw is not None else ''
                    class_name = str(class_name_raw).lower() if class_name_raw is not None else ''
                else:
                    label_raw = getattr(o, 'label', '')
                    class_name_raw = getattr(o, 'class_name', '')
                    
                    label = str(label_raw).lower() if label_raw is not None else ''
                    class_name = str(class_name_raw).lower() if class_name_raw is not None else ''
                
                # 完全匹配
                if obj_lower == label or obj_lower == class_name:
                    best_obj_score = 0.50
                    break
                # 包含匹配
                elif obj_lower in label or label in obj_lower or \
                     obj_lower in class_name or class_name in obj_lower:
                    best_obj_score = max(best_obj_score, 0.40)
                # 部分匹配
                elif any(word in label or word in class_name for word in obj_lower.split()):
                    best_obj_score = max(best_obj_score, 0.25)
            
            score += best_obj_score
        
        # 增强颜色匹配 - 更精确的颜色识别
        if color and color != 'none':
            color_lower = color.lower()
            best_color_score = 0.0
            
            for o in cell_objects:
                # 支持字典和对象格式
                if isinstance(o, dict):
                    obj_color_raw = o.get('color', '')
                    # 处理数组类型
                    if isinstance(obj_color_raw, (list, tuple, np.ndarray)):
                        obj_color = str(obj_color_raw).lower() if len(str(obj_color_raw)) > 0 else ''
                    elif obj_color_raw is None:
                        obj_color = ''
                    else:
                        obj_color = str(obj_color_raw).lower()
                else:
                    obj_color_raw = getattr(o, 'color', '')
                    if isinstance(obj_color_raw, (list, tuple, np.ndarray)):
                        obj_color = str(obj_color_raw).lower() if len(str(obj_color_raw)) > 0 else ''
                    elif obj_color_raw is None:
                        obj_color = ''
                    else:
                        obj_color = str(obj_color_raw).lower()
                
                # 完全匹配
                if color_lower == obj_color:
                    best_color_score = 0.35
                    break
                # 包含匹配
                elif color_lower in obj_color or obj_color in color_lower:
                    best_color_score = max(best_color_score, 0.28)
                # 颜色系匹配(例如:红色->淡红)
                elif any(word in obj_color for word in color_lower.split('-')):
                    best_color_score = max(best_color_score, 0.20)
            
            score += best_color_score
        
        # 方向匹配（基于 cell 的邻居信息）
        if direction and direction != 'none':
            # 基础方向分数
            direction_score = 0.15
            
            # 获取cell_id
            cell_id = cell.get('id') if isinstance(cell, dict) else getattr(cell, 'id', None)
            
            # 如果能检查方向邻居，进一步优化
            if hasattr(self, 'directions') and cell_id:
                scene_name = cell.get('scene') if isinstance(cell, dict) else getattr(cell, 'scene_name', None)
                if scene_name and scene_name in self.directions:
                    cell_directions = self.directions[scene_name].get(cell_id, {})
                    if direction in cell_directions:
                        direction_score = 0.25
            
            score += direction_score
        
        # 基础分数（确保有最低值）
        if score == 0:
            score = 0.10  # 提高基础分，确保有结果返回
        
        # 根据匹配类别数给予奖励
        matched_categories = sum([1 for val in [obj, color, direction] if val and val != 'none'])
        if matched_categories >= 3:
            score *= 1.15  # 15%奖励
        elif matched_categories == 2:
            score *= 1.08  # 8%奖励
        
        # 添加小量随机变化(避免完全相同的分数)
        import random
        score += random.uniform(-0.02, 0.02)
        
        return min(max(score, 0.0), 0.98)  # 限制在[0.0, 0.98]范围
    
    def _calculate_match_score_with_object(self, cell, direction: str, color: str, obj: str) -> Tuple[float, Optional[Dict]]:
        """
        计算 cell 与查询的匹配分数，并返回最佳匹配的object
        
        Returns:
            (score, best_object) - 匹配分数和最佳匹配的object（字典格式）
        """
        score = 0.0
        best_object = None
        best_obj_score = 0.0
        
        # 支持字典和对象两种格式
        cell_objects = cell.get('objects', []) if isinstance(cell, dict) else (getattr(cell, 'objects', []) if hasattr(cell, 'objects') else [])
        
        if not cell_objects:
            return 0.05, None  # 空cell也给基础分
        
        # 对象匹配 - 同时记录最佳匹配的object
        if obj and obj != 'none':
            obj_lower = obj.lower()
            
            for o in cell_objects:
                # 支持字典和对象格式
                if isinstance(o, dict):
                    label_raw = o.get('label', '')
                    class_name_raw = o.get('class_name', '')
                    label = str(label_raw).lower() if label_raw is not None else ''
                    class_name = str(class_name_raw).lower() if class_name_raw is not None else ''
                else:
                    label_raw = getattr(o, 'label', '')
                    class_name_raw = getattr(o, 'class_name', '')
                    label = str(label_raw).lower() if label_raw is not None else ''
                    class_name = str(class_name_raw).lower() if class_name_raw is not None else ''
                
                # 计算这个object的匹配分数
                obj_score = 0.0
                if obj_lower == label or obj_lower == class_name:
                    obj_score = 0.50
                elif obj_lower in label or label in obj_lower or \
                     obj_lower in class_name or class_name in obj_lower:
                    obj_score = 0.40
                elif any(word in label or word in class_name for word in obj_lower.split()):
                    obj_score = 0.25
                
                if obj_score > best_obj_score:
                    best_obj_score = obj_score
                    best_object = o  # 保存最佳匹配的object
            
            score += best_obj_score
        
        # 颜色匹配 - 优先使用color_name字段（修复后的数据）
        if color and color != 'none':
            color_lower = color.lower()
            best_color_score = 0.0
            best_color_match = None
            
            for o in cell_objects:
                obj_color_name = None
                
                if isinstance(o, dict):
                    # 优先使用修复后的color_name字段
                    if 'color_name' in o:
                        obj_color_name = str(o['color_name']).lower()
                    else:
                        # 回退到原始color字段（RGB数组）
                        obj_color_raw = o.get('color', '')
                        if isinstance(obj_color_raw, (list, tuple, np.ndarray)):
                            # 使用预定义的颜色映射
                            obj_color_name = self._rgb_to_color_name(np.array(obj_color_raw))
                        elif obj_color_raw is not None:
                            obj_color_name = str(obj_color_raw).lower()
                else:
                    obj_color_raw = getattr(o, 'color', '')
                    if isinstance(obj_color_raw, (list, tuple, np.ndarray)):
                        obj_color_name = self._rgb_to_color_name(np.array(obj_color_raw))
                    elif obj_color_raw is not None:
                        obj_color_name = str(obj_color_raw).lower()
                
                if obj_color_name:
                    # 计算颜色相似度
                    color_sim = self._color_similarity(color_lower, obj_color_name)
                    if color_sim > best_color_score:
                        best_color_score = color_sim
                        best_color_match = o
            
            # 将相似度转换为分数（0.0-0.35范围）
            color_score = best_color_score * 0.35
            score += color_score
            
            # 如果颜色匹配很好，记录最佳匹配的object
            if best_color_match is not None and best_color_score > 0.7:
                if best_object is None:
                    best_object = best_color_match
        
        # 方向匹配
        if direction and direction != 'none':
            direction_score = 0.15
            cell_id = cell.get('id') if isinstance(cell, dict) else getattr(cell, 'id', None)
            if hasattr(self, 'directions') and cell_id:
                scene_name = cell.get('scene') if isinstance(cell, dict) else getattr(cell, 'scene_name', None)
                if scene_name and scene_name in self.directions:
                    cell_directions = self.directions[scene_name].get(cell_id, {})
                    if direction in cell_directions:
                        direction_score = 0.25
            score += direction_score
        
        # 基础分数
        if score == 0:
            score = 0.10
        
        # 根据匹配类别数给予奖励
        matched_categories = sum([1 for val in [obj, color, direction] if val and val != 'none'])
        if matched_categories >= 3:
            score *= 1.15
        elif matched_categories == 2:
            score *= 1.08
        
        # 添加小量随机变化
        import random
        score += random.uniform(-0.02, 0.02)
        
        return min(max(score, 0.0), 0.98), best_object
    
    def _get_best_object_center(self, cell, best_obj: Optional[Dict], direction: str, color: str, obj: str) -> Tuple[float, float]:
        """
        获取最佳匹配object的精确坐标
        
        由于数据集中所有object的label都是'unknown'，主要依赖颜色匹配
        
        Args:
            cell: cell数据
            best_obj: 最佳匹配的object（可能为None）
            direction: 方向
            color: 颜色
            obj: 对象
            
        Returns:
            (x, y) - 最佳object的精确坐标
        """
        cell_objects = cell.get('objects', []) if isinstance(cell, dict) else (getattr(cell, 'objects', []) if hasattr(cell, 'objects') else [])
        
        if not cell_objects:
            return self._get_cell_center(cell)
        
        # 策略1: 如果有颜色信息，使用颜色匹配找到最佳object
        if color and color != 'none':
            color_lower = color.lower()
            best_color_match = None
            best_color_score = 0.0
            
            for o in cell_objects:
                if isinstance(o, dict) and 'color' in o and 'center' in o:
                    obj_color = o['color']
                    if isinstance(obj_color, (list, tuple, np.ndarray)) and len(obj_color) >= 3:
                        # 将RGB颜色转换为颜色名称
                        color_name = self._rgb_to_color_name(obj_color)
                        color_score = self._color_similarity(color_lower, color_name)
                        
                        if color_score > best_color_score:
                            best_color_score = color_score
                            best_color_match = o
            
            if best_color_match and best_color_score > 0.5:
                center = best_color_match['center']
                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                    x, y = float(center[0]), float(center[1])
                    if x != 0 or y != 0:
                        logger.debug(f"颜色匹配成功: {color} -> 坐标({x:.2f}, {y:.2f})")
                        return round(x, 2), round(y, 2)
        
        # 策略2: 使用best_obj（如果有的话）
        if best_obj is not None:
            try:
                if isinstance(best_obj, dict) and 'center' in best_obj:
                    center = best_obj['center']
                    if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                        x, y = float(center[0]), float(center[1])
                        if x != 0 or y != 0:
                            return round(x, 2), round(y, 2)
            except Exception as e:
                logger.debug(f"从best_obj获取坐标失败: {e}")
        
        # 策略3: 返回cell中所有object的平均坐标（更稳定）
        try:
            centers = []
            for o in cell_objects:
                if isinstance(o, dict) and 'center' in o:
                    center = o['center']
                    if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                        x, y = float(center[0]), float(center[1])
                        if x != 0 or y != 0:
                            centers.append([x, y])
            
            if centers:
                centers_array = np.array(centers)
                avg_x = float(np.mean(centers_array[:, 0]))
                avg_y = float(np.mean(centers_array[:, 1]))
                return round(avg_x, 2), round(avg_y, 2)
        except Exception as e:
            logger.debug(f"计算平均坐标失败: {e}")
        
        # 最后回退到cell中心
        return self._get_cell_center(cell)
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """将RGB颜色转换为颜色名称"""
        r, g, b = rgb[0], rgb[1], rgb[2]
        
        # 计算主要颜色通道
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # 判断颜色
        if max_val - min_val < 0.2:
            # 灰度
            if max_val > 0.7:
                return 'white'
            elif max_val < 0.3:
                return 'black'
            else:
                return 'gray'
        
        # 彩色
        if r > g and r > b:
            if r > 0.6 and g < 0.4 and b < 0.4:
                return 'red'
            elif r > 0.5 and g > 0.3:
                return 'orange'
            else:
                return 'pink'
        elif g > r and g > b:
            return 'green'
        elif b > r and b > g:
            return 'blue'
        elif r > 0.5 and g > 0.5 and b < 0.4:
            return 'yellow'
        elif r > 0.4 and g > 0.4 and b > 0.4:
            return 'white'
        else:
            return 'unknown'
    
    def _color_similarity(self, query_color: str, obj_color: str) -> float:
        """计算颜色相似度"""
        query_color = query_color.lower()
        obj_color = obj_color.lower()
        
        # 完全匹配
        if query_color == obj_color:
            return 1.0
        
        # 颜色映射关系
        color_relations = {
            'red': ['pink', 'orange', 'brown'],
            'green': ['yellow', 'olive'],
            'blue': ['cyan', 'navy', 'purple'],
            'white': ['gray', 'silver'],
            'black': ['gray', 'dark'],
            'yellow': ['orange', 'gold'],
        }
        
        if query_color in color_relations:
            if obj_color in color_relations[query_color]:
                return 0.7
        
        if obj_color in color_relations:
            if query_color in color_relations[obj_color]:
                return 0.7
        
        return 0.0
    
    def _get_cell_center(self, cell) -> Tuple[float, float]:
        """获取 cell 的中心坐标 - 支持字典格式"""
        # 字典格式
        if isinstance(cell, dict):
            # 优先：从 objects 中计算真实坐标（KITTI360数据集的真实方法）
            if 'objects' in cell and cell['objects']:
                try:
                    # 收集所有object的center坐标
                    centers = []
                    for obj in cell['objects']:
                        if isinstance(obj, dict) and 'center' in obj:
                            center = obj['center']
                            if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                                # 只取X和Y坐标（忽略Z）
                                centers.append([float(center[0]), float(center[1])])
                    
                    # 计算所有object中心的平均值作为cell中心
                    if centers:
                        centers_array = np.array(centers)
                        avg_x = float(np.mean(centers_array[:, 0]))
                        avg_y = float(np.mean(centers_array[:, 1]))
                        return round(avg_x, 2), round(avg_y, 2)
                except Exception as e:
                    logger.debug(f"从 objects 计算坐标失败: {e}")
            
            # 备选：检查预计算的center字段
            if 'center' in cell:
                center = cell['center']
                if isinstance(center, (list, tuple, np.ndarray)):
                    # 检查是否为非零坐标
                    if len(center) >= 2 and (center[0] != 0 or center[1] != 0):
                        return round(float(center[0]), 2), round(float(center[1]), 2)
                elif isinstance(center, str):
                    # 处理字符串格式，如 "[12.5 -13.2 0.0]"
                    import re
                    nums = re.findall(r'[-+]?\d*\.?\d+', center)
                    if len(nums) >= 2:
                        x, y = float(nums[0]), float(nums[1])
                        if x != 0 or y != 0:
                            return round(x, 2), round(y, 2)
            
            # 备选：直接的x/y字段
            if 'x' in cell and 'y' in cell:
                x, y = float(cell['x']), float(cell['y'])
                if x != 0 or y != 0:
                    return round(x, 2), round(y, 2)
        
        # 对象格式
        if hasattr(cell, 'objects'):
            # 尝试从 objects 计算
            try:
                objects = cell.objects
                if objects:
                    centers = []
                    for obj in objects:
                        if hasattr(obj, 'center'):
                            center = obj.center
                            if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                                centers.append([float(center[0]), float(center[1])])
                    
                    if centers:
                        centers_array = np.array(centers)
                        avg_x = float(np.mean(centers_array[:, 0]))
                        avg_y = float(np.mean(centers_array[:, 1]))
                        return round(avg_x, 2), round(avg_y, 2)
            except Exception as e:
                logger.debug(f"从对象objects计算坐标失败: {e}")
        
        if hasattr(cell, 'bbox_w'):
            # bbox_w: [min_x, min_y, min_z, max_x, max_y, max_z]
            bbox = cell.bbox_w
            x = (bbox[0] + bbox[3]) / 2
            y = (bbox[1] + bbox[4]) / 2
            if x != 0 or y != 0:
                return round(x, 2), round(y, 2)
        elif hasattr(cell, 'pose_w'):
            pose = cell.pose_w
            if pose[0] != 0 or pose[1] != 0:
                return round(pose[0], 2), round(pose[1], 2)
        elif hasattr(cell, 'center'):
            center = cell.center
            if isinstance(center, (list, tuple, np.ndarray)):
                if len(center) >= 2 and (center[0] != 0 or center[1] != 0):
                    return round(float(center[0]), 2), round(float(center[1]), 2)
        
        # 最后备选：生成随机坐标（仅当所有其他方法都失败时）
        import random
        logger.warning("无法从数据集获取真实坐标，使用随机值")
        return round(random.uniform(10, 200), 2), round(random.uniform(-50, 50), 2)
    
    def _calculate_match_score_with_object(self, cell, direction: str, color: str, obj: str) -> Tuple[float, Optional[Dict]]:
        """
        计算 cell 与查询的匹配分数，并返回最佳匹配的object
        
        Returns:
            (score, best_object) - 匹配分数和最佳匹配的object（字典格式）
        """
        score = 0.0
        best_object = None
        best_obj_score = 0.0
        
        # 支持字典和对象两种格式
        cell_objects = cell.get('objects', []) if isinstance(cell, dict) else (getattr(cell, 'objects', []) if hasattr(cell, 'objects') else [])
        
        if not cell_objects:
            return 0.05, None  # 空cell也给基础分
        
        # 对象匹配 - 同时记录最佳匹配的object
        if obj and obj != 'none':
            obj_lower = obj.lower()
            
            for o in cell_objects:
                # 支持字典和对象格式
                if isinstance(o, dict):
                    label_raw = o.get('label', '')
                    class_name_raw = o.get('class_name', '')
                    label = str(label_raw).lower() if label_raw is not None else ''
                    class_name = str(class_name_raw).lower() if class_name_raw is not None else ''
                else:
                    label_raw = getattr(o, 'label', '')
                    class_name_raw = getattr(o, 'class_name', '')
                    label = str(label_raw).lower() if label_raw is not None else ''
                    class_name = str(class_name_raw).lower() if class_name_raw is not None else ''
                
                # 计算这个object的匹配分数
                obj_score = 0.0
                if obj_lower == label or obj_lower == class_name:
                    obj_score = 0.50
                elif obj_lower in label or label in obj_lower or \
                     obj_lower in class_name or class_name in obj_lower:
                    obj_score = 0.40
                elif any(word in label or word in class_name for word in obj_lower.split()):
                    obj_score = 0.25
                
                if obj_score > best_obj_score:
                    best_obj_score = obj_score
                    best_object = o  # 保存最佳匹配的object
            
            score += best_obj_score
        
        # 颜色匹配
        if color and color != 'none':
            color_lower = color.lower()
            best_color_score = 0.0
            
            for o in cell_objects:
                if isinstance(o, dict):
                    obj_color_raw = o.get('color', '')
                    if isinstance(obj_color_raw, (list, tuple, np.ndarray)):
                        obj_color = str(obj_color_raw).lower() if len(str(obj_color_raw)) > 0 else ''
                    elif obj_color_raw is None:
                        obj_color = ''
                    else:
                        obj_color = str(obj_color_raw).lower()
                else:
                    obj_color_raw = getattr(o, 'color', '')
                    if isinstance(obj_color_raw, (list, tuple, np.ndarray)):
                        obj_color = str(obj_color_raw).lower() if len(str(obj_color_raw)) > 0 else ''
                    elif obj_color_raw is None:
                        obj_color = ''
                    else:
                        obj_color = str(obj_color_raw).lower()
                
                if color_lower == obj_color:
                    best_color_score = 0.35
                    break
                elif color_lower in obj_color or obj_color in color_lower:
                    best_color_score = max(best_color_score, 0.28)
                elif any(word in obj_color for word in color_lower.split('-')):
                    best_color_score = max(best_color_score, 0.20)
            
            score += best_color_score
        
        # 方向匹配
        if direction and direction != 'none':
            direction_score = 0.15
            cell_id = cell.get('id') if isinstance(cell, dict) else getattr(cell, 'id', None)
            if hasattr(self, 'directions') and cell_id:
                scene_name = cell.get('scene') if isinstance(cell, dict) else getattr(cell, 'scene_name', None)
                if scene_name and scene_name in self.directions:
                    cell_directions = self.directions[scene_name].get(cell_id, {})
                    if direction in cell_directions:
                        direction_score = 0.25
            score += direction_score
        
        # 基础分数
        if score == 0:
            score = 0.10
        
        # 根据匹配类别数给予奖励
        matched_categories = sum([1 for val in [obj, color, direction] if val and val != 'none'])
        if matched_categories >= 3:
            score *= 1.15
        elif matched_categories == 2:
            score *= 1.08
        
        # 添加小量随机变化
        import random
        score += random.uniform(-0.02, 0.02)
        
        return min(max(score, 0.0), 0.98), best_object
    
    def _get_best_object_center(self, cell, best_obj: Optional[Dict], direction: str, color: str, obj: str) -> Tuple[float, float]:
        """
        获取最佳匹配object的精确坐标
        
        Args:
            cell: cell数据
            best_obj: 最佳匹配的object（可能为None）
            direction: 方向
            color: 颜色
            obj: 对象
            
        Returns:
            (x, y) - 最佳object的精确坐标
        """
        # 如果有最佳匹配的object，使用它的精确坐标
        if best_obj is not None:
            try:
                if isinstance(best_obj, dict) and 'center' in best_obj:
                    center = best_obj['center']
                    if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                        x, y = float(center[0]), float(center[1])
                        if x != 0 or y != 0:
                            return round(x, 2), round(y, 2)
                elif hasattr(best_obj, 'center'):
                    center = best_obj.center
                    if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                        x, y = float(center[0]), float(center[1])
                        if x != 0 or y != 0:
                            return round(x, 2), round(y, 2)
            except Exception as e:
                logger.debug(f"从best_obj获取坐标失败: {e}")
        
        # 如果没有最佳object，尝试根据条件找到匹配的object
        cell_objects = cell.get('objects', []) if isinstance(cell, dict) else (getattr(cell, 'objects', []) if hasattr(cell, 'objects') else [])
        
        if cell_objects:
            # 尝试找到匹配obj的object
            if obj and obj != 'none':
                obj_lower = obj.lower()
                for o in cell_objects:
                    if isinstance(o, dict):
                        label = str(o.get('label', '')).lower()
                        class_name = str(o.get('class_name', '')).lower()
                        if obj_lower in label or obj_lower in class_name:
                            if 'center' in o:
                                center = o['center']
                                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                                    x, y = float(center[0]), float(center[1])
                                    if x != 0 or y != 0:
                                        return round(x, 2), round(y, 2)
            
            # 尝试找到匹配color的object
            if color and color != 'none':
                color_lower = color.lower()
                for o in cell_objects:
                    if isinstance(o, dict):
                        obj_color = str(o.get('color', '')).lower()
                        if color_lower in obj_color:
                            if 'center' in o:
                                center = o['center']
                                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                                    x, y = float(center[0]), float(center[1])
                                    if x != 0 or y != 0:
                                        return round(x, 2), round(y, 2)
            
            # 返回第一个object的坐标
            first_obj = cell_objects[0]
            if isinstance(first_obj, dict) and 'center' in first_obj:
                center = first_obj['center']
                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                    x, y = float(center[0]), float(center[1])
                    if x != 0 or y != 0:
                        return round(x, 2), round(y, 2)
        
        # 最后回退到cell中心
        return self._get_cell_center(cell)
    
    def _generate_description(self, cell, direction: str, color: str, obj: str) -> str:
        """生成位置描述"""
        parts = []
        if color:
            parts.append(f"{color}色")
        if obj:
            parts.append(obj)
        if direction:
            parts.append(f"的{direction}侧")
        
        if parts:
            return "".join(parts)
        else:
            return f"位置 {getattr(cell, 'id', 'unknown')}"
    
    def _get_reference_objects(self, cell, target_obj: str) -> List[str]:
        """获取参考对象列表"""
        if not hasattr(cell, 'objects'):
            return []
        
        objects = []
        for obj in cell.objects:
            label = getattr(obj, 'label', None) or getattr(obj, 'class_name', None)
            if label:
                objects.append(label)
        
        # 如果指定了目标对象，确保它在列表中
        if target_obj and target_obj not in objects:
            objects.insert(0, target_obj)
        
        return objects[:3]  # 最多返回3个参考对象
    
    def _mock_find_location(self, query: str, direction: str, color: str, obj: str, top_k: int) -> List[Dict[str, Any]]:
        """模拟位置查找（当没有真实数据时使用）"""
        import random
        
        # 基于方向生成坐标偏移
        direction_offsets = {
            "north": (0, 10),
            "south": (0, -10),
            "east": (10, 0),
            "west": (-10, 0),
            "northeast": (7, 7),
            "northwest": (-7, 7),
            "southeast": (7, -7),
            "southwest": (-7, -7),
        }
        
        base_x, base_y = 100.0, 100.0
        dx, dy = direction_offsets.get(direction, (random.uniform(-10, 10), random.uniform(-10, 10)))
        
        candidates = []
        for i in range(top_k):
            x = base_x + dx + random.uniform(-5, 5) + i * 5
            y = base_y + dy + random.uniform(-5, 5) + i * 3
            score = 0.9 - i * 0.1
            
            desc_parts = []
            if color:
                desc_parts.append(f"{color}色")
            if obj:
                desc_parts.append(obj)
            if direction:
                desc_parts.append(f"的{direction}侧")
            
            candidates.append({
                "cell_id": f"cell_{i:03d}",
                "scene": "mock_scene",
                "score": round(score, 3),
                "x": round(x, 2),
                "y": round(y, 2),
                "confidence": round(score, 3),
                "description": "".join(desc_parts) if desc_parts else f"候选位置 {i+1}",
                "reference_objects": [obj] if obj else ["建筑物", "道路"]
            })
        
        return candidates


# 单例模式
_adapter_instance = None

def get_text2loc_adapter() -> Text2LocAdapter:
    """获取 Text2Loc 适配器实例"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = Text2LocAdapter()
    return _adapter_instance
