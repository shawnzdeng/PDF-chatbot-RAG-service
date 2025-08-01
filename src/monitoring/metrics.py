"""
Metrics collection for RAG system monitoring
Provides Prometheus-compatible metrics for system performance tracking
"""

import time
from typing import Dict, Any
from datetime import datetime
import threading
from collections import defaultdict, deque
import psutil
import logging

logger = logging.getLogger(__name__)

class RAGMetrics:
    """Collects and exposes metrics for the RAG system"""
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.request_duration = defaultdict(list)
        self.error_count = defaultdict(int)
        self.active_sessions = set()
        self.query_embeddings_count = 0
        self.retrieval_count = 0
        self.response_generation_count = 0
        self.qdrant_operations = defaultdict(int)
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        self._lock = threading.Lock()
        
    def record_request(self, endpoint: str = "default"):
        """Record a new request"""
        with self._lock:
            self.request_count[endpoint] += 1
    
    def record_request_duration(self, duration: float, endpoint: str = "default"):
        """Record request duration in seconds"""
        with self._lock:
            self.request_duration[endpoint].append(duration)
            self.response_times.append(duration)
            # Keep only last 100 durations per endpoint
            if len(self.request_duration[endpoint]) > 100:
                self.request_duration[endpoint] = self.request_duration[endpoint][-100:]
    
    def record_error(self, error_type: str = "general"):
        """Record an error occurrence"""
        with self._lock:
            self.error_count[error_type] += 1
    
    def record_session(self, session_id: str):
        """Record an active session"""
        with self._lock:
            self.active_sessions.add(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session"""
        with self._lock:
            self.active_sessions.discard(session_id)
    
    def record_query_embedding(self):
        """Record a query embedding operation"""
        with self._lock:
            self.query_embeddings_count += 1
    
    def record_retrieval(self):
        """Record a document retrieval operation"""
        with self._lock:
            self.retrieval_count += 1
    
    def record_response_generation(self):
        """Record a response generation"""
        with self._lock:
            self.response_generation_count += 1
    
    def record_qdrant_operation(self, operation: str):
        """Record a Qdrant database operation"""
        with self._lock:
            self.qdrant_operations[operation] += 1
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / 1024 / 1024,
                "memory_total_mb": memory.total / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / 1024 / 1024 / 1024,
                "disk_total_gb": disk.total / 1024 / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics string"""
        with self._lock:
            metrics = []
            
            # Request count metrics
            for endpoint, count in self.request_count.items():
                metrics.append(f'rag_requests_total{{endpoint="{endpoint}"}} {count}')
            
            # Error count metrics
            for error_type, count in self.error_count.items():
                metrics.append(f'rag_errors_total{{type="{error_type}"}} {count}')
            
            # Active sessions
            metrics.append(f'rag_active_sessions {len(self.active_sessions)}')
            
            # Operation counts
            metrics.append(f'rag_query_embeddings_total {self.query_embeddings_count}')
            metrics.append(f'rag_retrievals_total {self.retrieval_count}')
            metrics.append(f'rag_response_generations_total {self.response_generation_count}')
            
            # Qdrant operations
            for operation, count in self.qdrant_operations.items():
                metrics.append(f'rag_qdrant_operations_total{{operation="{operation}"}} {count}')
            
            # Response time metrics
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                max_response_time = max(self.response_times)
                min_response_time = min(self.response_times)
                
                metrics.append(f'rag_response_time_seconds_avg {avg_response_time:.4f}')
                metrics.append(f'rag_response_time_seconds_max {max_response_time:.4f}')
                metrics.append(f'rag_response_time_seconds_min {min_response_time:.4f}')
            
            # System metrics
            system_metrics = self.get_system_metrics()
            for metric, value in system_metrics.items():
                metrics.append(f'rag_system_{metric} {value}')
            
            # Add timestamp
            metrics.append(f'# HELP rag_metrics_last_updated_timestamp Last update timestamp')
            metrics.append(f'rag_metrics_last_updated_timestamp {time.time()}')
            
            return '\n'.join(metrics)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for health checks"""
        try:
            system_metrics = self.get_system_metrics()
            
            # Basic health indicators
            is_healthy = True
            issues = []
            
            # Check CPU usage
            if system_metrics.get('cpu_percent', 0) > 90:
                is_healthy = False
                issues.append("High CPU usage")
            
            # Check memory usage
            if system_metrics.get('memory_percent', 0) > 90:
                is_healthy = False
                issues.append("High memory usage")
            
            # Check disk usage
            if system_metrics.get('disk_percent', 0) > 90:
                is_healthy = False
                issues.append("High disk usage")
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "issues": issues,
                "metrics": system_metrics,
                "active_sessions": len(self.active_sessions),
                "total_requests": sum(self.request_count.values()),
                "total_errors": sum(self.error_count.values())
            }
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

# Global metrics instance
rag_metrics = RAGMetrics()

class MetricsContextManager:
    """Context manager for timing operations"""
    
    def __init__(self, metrics: RAGMetrics, endpoint: str = "default"):
        self.metrics = metrics
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.metrics.record_request(self.endpoint)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_request_duration(duration, self.endpoint)
        
        if exc_type is not None:
            self.metrics.record_error(exc_type.__name__)

def time_operation(endpoint: str = "default"):
    """Decorator for timing operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with MetricsContextManager(rag_metrics, endpoint):
                return func(*args, **kwargs)
        return wrapper
    return decorator
