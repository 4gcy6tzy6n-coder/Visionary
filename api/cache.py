"""
Text2Loc API 缓存机制
提供查询结果缓存，加速重复查询
"""

import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class QueryCache:
    """
    查询结果缓存
    
    特性：
    - LRU淘汰策略
    - 线程安全
    - 持久化支持
    - TTL过期机制
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: "OrderedDict[str, Tuple[float, Dict]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
        logger.info(f"查询缓存初始化完成: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            缓存键
        """
        # 标准化查询（去除多余空格，统一大小写）
        normalized = ' '.join(query.lower().split())
        
        # 包含关键参数
        key_parts = [normalized]
        if 'top_k' in kwargs:
            key_parts.append(f"k{kwargs['top_k']}")
        if 'enable_enhanced' in kwargs:
            key_parts.append(f"e{int(kwargs['enable_enhanced'])}")
        
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        获取缓存结果
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            缓存的结果或None
        """
        key = self._generate_key(query, **kwargs)
        
        with self.lock:
            if key in self.cache:
                timestamp, result = self.cache[key]
                
                # 检查是否过期
                if time.time() - timestamp < self.ttl:
                    # 移动到末尾（LRU）
                    self.cache.move_to_end(key)
                    self.hits += 1
                    logger.debug(f"缓存命中: {key[:8]}...")
                    return result
                else:
                    # 过期，删除
                    del self.cache[key]
            
            self.misses += 1
            logger.debug(f"缓存未命中: {key[:8]}...")
            return None
    
    def set(self, query: str, result: Dict[str, Any], **kwargs):
        """
        缓存结果
        
        Args:
            query: 查询文本
            result: 结果字典
            **kwargs: 其他参数
        """
        key = self._generate_key(query, **kwargs)
        
        with self.lock:
            # 如果已存在，先删除
            if key in self.cache:
                del self.cache[key]
            
            # 添加新条目
            self.cache[key] = (time.time(), result)
            
            # LRU淘汰
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            
            logger.debug(f"缓存写入: {key[:8]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计
        
        Returns:
            统计信息字典
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": self.ttl
        }
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("缓存已清空")
    
    def cleanup(self):
        """清理过期条目"""
        with self.lock:
            now = time.time()
            expired = []
            for key, (timestamp, _) in self.cache.items():
                if now - timestamp >= self.ttl:
                    expired.append(key)
            
            for key in expired:
                del self.cache[key]
            
            if expired:
                logger.info(f"清理了 {len(expired)} 个过期缓存条目")


class EnhancedText2LocAPI:
    """
    增强版Text2Loc API（带缓存）
    
    在原始Text2LocAPI基础上添加：
    - 查询结果缓存
    - 性能优化
    - 批量处理支持
    """
    
    def __init__(self, adapter=None, config=None, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        初始化增强API
        
        Args:
            adapter: Text2Loc适配器实例
            config: 配置
            cache_size: 缓存大小
            cache_ttl: 缓存过期时间（秒）
        """
        from api.text2loc_api import Text2LocAPI
        
        # 原始API
        self.api = Text2LocAPI(adapter=adapter, config=config)
        
        # 缓存
        self.cache = QueryCache(max_size=cache_size, ttl_seconds=cache_ttl)
        
        logger.info("EnhancedText2LocAPI 初始化完成")
    
    def process_query(self, request, use_cache: bool = True) -> Any:
        """
        处理查询（带缓存）
        
        Args:
            request: 查询请求
            use_cache: 是否使用缓存
            
        Returns:
            查询响应
        """
        # 尝试从缓存获取
        if use_cache:
            cache_key = self.cache._generate_key(
                request.query,
                top_k=request.top_k,
                enable_enhanced=request.enable_enhanced
            )
            
            cached = self.cache.get(
                request.query,
                top_k=request.top_k,
                enable_enhanced=request.enable_enhanced
            )
            
            if cached:
                logger.info(f"使用缓存结果: {cache_key[:8]}...")
                return cached
        
        # 执行查询
        response = self.api.process_query(request)
        
        # 缓存结果
        if use_cache and response.status == "success":
            self.cache.set(
                request.query,
                response,
                top_k=request.top_k,
                enable_enhanced=request.enable_enhanced
            )
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
    
    def __getattr__(self, name):
        """代理其他方法"""
        return getattr(self.api, name)


# 全局缓存实例
_global_cache: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryCache(max_size=1000, ttl_seconds=3600)
    return _global_cache


def init_cache(max_size: int = 1000, ttl_seconds: int = 3600) -> QueryCache:
    """初始化全局缓存"""
    global _global_cache
    _global_cache = QueryCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _global_cache
