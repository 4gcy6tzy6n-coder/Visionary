"""
监控系统模块

提供性能指标收集、错误跟踪和系统状态监控功能
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    module_name: str
    operation_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    error_count: int = 0
    last_called: datetime = field(default_factory=datetime.now)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        return self.avg_time_ms

    @property
    def error_rate(self) -> float:
        return self.error_count / self.call_count if self.call_count > 0 else 0.0


class MetricsCollector:
    """性能指标收集器"""

    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
        self._request_times: List[float] = []
        self._query_count = 0
        self._success_count = 0
        self._error_count = 0

    def record_operation(self, module: str, operation: str, duration_ms: float, 
                        success: bool = True, details: Optional[Dict] = None):
        """记录操作性能指标"""
        key = f"{module}:{operation}"
        
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = PerformanceMetrics(
                    module_name=module,
                    operation_name=operation
                )
            
            metrics = self._metrics[key]
            metrics.call_count += 1
            metrics.total_time_ms += duration_ms
            metrics.min_time_ms = min(metrics.min_time_ms, duration_ms)
            metrics.max_time_ms = max(metrics.max_time_ms, duration_ms)
            metrics.last_called = datetime.now()
            
            if not success:
                metrics.error_count += 1

        if module == "api" and operation == "process_query":
            self._request_times.append(duration_ms)
            self._query_count += 1
            if success:
                self._success_count += 1
            else:
                self._error_count += 1

    def get_module_metrics(self, module: str) -> List[PerformanceMetrics]:
        """获取指定模块的指标"""
        with self._lock:
            return [m for m in self._metrics.values() if m.module_name == module]

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            # 直接在锁内进行分组，不调用可能再次获取锁的方法
            from collections import defaultdict
            grouped = defaultdict(list)
            for metrics in self._metrics.values():
                grouped[metrics.module_name].append(metrics)
            
            modules_data = {}
            for module, ops in grouped.items():
                modules_data[module] = {
                    "operations": [
                        {
                            "name": m.operation_name,
                            "call_count": m.call_count,
                            "avg_time_ms": round(m.avg_time_ms, 3),
                            "min_time_ms": round(m.min_time_ms, 3) if m.min_time_ms != float('inf') else 0,
                            "max_time_ms": round(m.max_time_ms, 3),
                            "error_count": m.error_count,
                            "error_rate": round(m.error_rate, 4)
                        }
                        for m in ops
                    ]
                }
            
            # 预先计算summary数据
            total_calls = sum(m.call_count for m in self._metrics.values())
            total_errors = sum(m.error_count for m in self._metrics.values())
            avg_query_time = sum(self._request_times) / len(self._request_times) if self._request_times else 0
            
            summary = {
                "total_operations": total_calls,
                "total_errors": total_errors,
                "overall_error_rate": round(total_errors / total_calls, 4) if total_calls > 0 else 0,
                "query_count": self._query_count,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "success_rate": round(self._success_count / self._query_count, 4) if self._query_count > 0 else 0,
                "avg_query_time_ms": round(avg_query_time, 3),
                "p95_query_time_ms": self._calculate_percentile(95),
                "p99_query_time_ms": self._calculate_percentile(99)
            }
            
            return {
                "modules": modules_data,
                "summary": summary
            }

    def _get_modules_grouped(self) -> Dict[str, List[PerformanceMetrics]]:
        """按模块分组指标"""
        grouped = defaultdict(list)
        with self._lock:
            for metrics in self._metrics.values():
                grouped[metrics.module_name].append(metrics)
        return grouped

    def get_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        with self._lock:
            total_calls = sum(m.call_count for m in self._metrics.values())
            total_errors = sum(m.error_count for m in self._metrics.values())
            avg_query_time = sum(self._request_times) / len(self._request_times) if self._request_times else 0
            
            return {
                "total_operations": total_calls,
                "total_errors": total_errors,
                "overall_error_rate": round(total_errors / total_calls, 4) if total_calls > 0 else 0,
                "query_count": self._query_count,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "success_rate": round(self._success_count / self._query_count, 4) if self._query_count > 0 else 0,
                "avg_query_time_ms": round(avg_query_time, 3),
                "p95_query_time_ms": self._calculate_percentile(95),
                "p99_query_time_ms": self._calculate_percentile(99)
            }

    def _calculate_percentile(self, percentile: int) -> float:
        """计算响应时间百分位"""
        if not self._request_times:
            return 0
        sorted_times = sorted(self._request_times)
        index = int(len(sorted_times) * percentile / 100)
        return round(sorted_times[min(index, len(sorted_times) - 1)], 3)


class ErrorTracker:
    """错误跟踪器"""

    def __init__(self, max_errors: int = 1000):
        self._errors: List[Dict] = []
        self._max_errors = max_errors
        self._lock = threading.Lock()

    def record_error(self, error_type: str, message: str, module: str, 
                    details: Optional[Dict] = None):
        """记录错误"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "module": module,
            "details": details or {}
        }
        
        with self._lock:
            self._errors.append(error_record)
            if len(self._errors) > self._max_errors:
                self._errors.pop(0)

    def get_recent_errors(self, limit: int = 50) -> List[Dict]:
        """获取最近的错误"""
        with self._lock:
            return list(self._errors[-limit:])

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        with self._lock:
            error_counts = {}
            for error in self._errors:
                error_type = error["type"]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            return {
                "total_errors": len(self._errors),
                "error_types": error_counts,
                "recent_errors": list(self._errors[-10:])
            }


class UserFeedbackCollector:
    """用户反馈收集器"""

    def __init__(self, max_feedback: int = 500):
        self._feedback: List[Dict] = []
        self._max_feedback = max_feedback
        self._lock = threading.Lock()

    def add_feedback(self, query_id: str, rating: int, comment: Optional[str],
                    query_text: str, result_quality: str):
        """添加用户反馈"""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "rating": rating,
            "comment": comment,
            "query_text": query_text,
            "result_quality": result_quality
        }
        
        with self._lock:
            self._feedback.append(feedback)
            if len(self._feedback) > self._max_feedback:
                self._feedback.pop(0)

    def get_feedback_stats(self) -> Dict[str, Any]:
        """获取反馈统计"""
        with self._lock:
            if not self._feedback:
                return {"total": 0, "average_rating": 0, "distribution": {}}
            
            ratings = [f["rating"] for f in self._feedback]
            distribution = {}
            for r in range(1, 6):
                distribution[r] = ratings.count(r)
            
            return {
                "total": len(self._feedback),
                "average_rating": round(sum(ratings) / len(ratings), 2),
                "distribution": distribution,
                "recent_feedback": list(self._feedback[-5:])
            }


class SystemMonitor:
    """系统监控器"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.error_tracker = ErrorTracker()
        self.feedback_collector = UserFeedbackCollector()
        self._start_time = datetime.now()
        self._status = "healthy"

    @property
    def uptime_seconds(self) -> float:
        """系统运行时间（秒）"""
        return (datetime.now() - self._start_time).total_seconds()

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": self._status,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics_collector.get_all_metrics(),
            "errors": self.error_tracker.get_error_summary(),
            "feedback": self.feedback_collector.get_feedback_stats()
        }

    def record_query(self, query_id: str, duration_ms: float, success: bool,
                    details: Optional[Dict] = None):
        """记录查询"""
        self.metrics_collector.record_operation(
            module="api",
            operation="process_query",
            duration_ms=duration_ms,
            success=success,
            details={"query_id": query_id} if details is None else details
        )

    def record_module_operation(self, module: str, operation: str, 
                               duration_ms: float, success: bool = True):
        """记录模块操作"""
        self.metrics_collector.record_operation(
            module=module,
            operation=operation,
            duration_ms=duration_ms,
            success=success
        )

    def track_error(self, error_type: str, message: str, module: str,
                   details: Optional[Dict] = None):
        """跟踪错误"""
        self.error_tracker.record_error(error_type, message, module, details)

    def collect_feedback(self, query_id: str, rating: int, comment: Optional[str],
                        query_text: str, result_quality: str):
        """收集用户反馈"""
        self.feedback_collector.add_feedback(
            query_id, rating, comment, query_text, result_quality
        )


# 全局监控器实例（延迟初始化避免导入问题）
_monitor_instance = None

def get_monitor() -> SystemMonitor:
    """获取全局监控器实例"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """配置日志系统"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"日志系统初始化完成，级别: {log_level}")


def monitor_operation(module: str, operation: str):
    """监控操作装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_msg = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_module_operation(module, operation, duration_ms, success)
                if not success:
                    monitor.track_error(
                        error_type=type(e).__name__,
                        message=error_msg,
                        module=module
                    )
        
        return wrapper
    return decorator
