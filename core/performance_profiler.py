import time
import psutil
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import logging


@dataclass
class OperationMetrics:
    """Metrics for a single operation execution."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    cpu_before: float
    cpu_after: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    thread_id: int
    timestamp: str


class PerformanceProfiler:
    """Profiles individual operations and their resource consumption."""
    
    def __init__(self, enabled: bool = True, log_file: str = "performance_profile.json"):
        """
        Initialize the performance profiler.
        
        Args:
            enabled: Whether profiling is enabled
            log_file: File to save performance data
        """
        self.enabled = enabled
        self.log_file = log_file
        self.operations: List[OperationMetrics] = []
        self.operation_stack: Dict[int, Dict] = {}  # thread_id -> operation info
        self.process = psutil.Process()
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def start_operation(self, operation_name: str) -> Optional[int]:
        """
        Start profiling an operation.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Returns:
            Operation ID if profiling is enabled, None otherwise
        """
        if not self.enabled:
            return None
            
        thread_id = threading.get_ident()
        current_time = time.time()
        
        # Get initial resource metrics
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Error getting initial metrics: {e}")
            cpu_percent = 0.0
            memory_mb = 0.0
        
        with self.lock:
            self.operation_stack[thread_id] = {
                'operation_name': operation_name,
                'start_time': current_time,
                'cpu_before': cpu_percent,
                'memory_before_mb': memory_mb,
                'memory_peak_mb': memory_mb
            }
            
        return thread_id
        
    def end_operation(self, operation_id: Optional[int]) -> Optional[OperationMetrics]:
        """
        End profiling an operation and record metrics.
        
        Args:
            operation_id: ID returned by start_operation
            
        Returns:
            OperationMetrics if successful, None otherwise
        """
        if not self.enabled or operation_id is None:
            return None
            
        thread_id = operation_id
        current_time = time.time()
        
        with self.lock:
            if thread_id not in self.operation_stack:
                return None
                
            operation_info = self.operation_stack.pop(thread_id)
            
        # Get final resource metrics
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Error getting final metrics: {e}")
            cpu_percent = 0.0
            memory_mb = operation_info['memory_before_mb']
        
        # Calculate metrics
        duration_ms = (current_time - operation_info['start_time']) * 1000
        memory_peak_mb = max(memory_mb, operation_info['memory_peak_mb'])
        
        metrics = OperationMetrics(
            operation_name=operation_info['operation_name'],
            start_time=operation_info['start_time'],
            end_time=current_time,
            duration_ms=duration_ms,
            cpu_before=operation_info['cpu_before'],
            cpu_after=cpu_percent,
            memory_before_mb=operation_info['memory_before_mb'],
            memory_after_mb=memory_mb,
            memory_peak_mb=memory_peak_mb,
            thread_id=thread_id,
            timestamp=datetime.now().isoformat()
        )
        
        with self.lock:
            self.operations.append(metrics)
            
        return metrics
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling operations.
        
        Usage:
            with profiler.profile_operation("my_operation"):
                # Your code here
                pass
        """
        operation_id = self.start_operation(operation_name)
        try:
            yield
        finally:
            self.end_operation(operation_id)
            
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation_name: Specific operation name, or None for all operations
            
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            if operation_name:
                ops = [op for op in self.operations if op.operation_name == operation_name]
            else:
                ops = self.operations
                
        if not ops:
            return {}
            
        durations = [op.duration_ms for op in ops]
        memory_deltas = [op.memory_after_mb - op.memory_before_mb for op in ops]
        cpu_deltas = [op.cpu_after - op.cpu_before for op in ops]
        
        return {
            'operation_name': operation_name or 'all',
            'total_executions': len(ops),
            'duration_ms': {
                'avg': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'total': sum(durations)
            },
            'memory_delta_mb': {
                'avg': sum(memory_deltas) / len(memory_deltas),
                'min': min(memory_deltas),
                'max': max(memory_deltas),
                'total': sum(memory_deltas)
            },
            'cpu_delta_percent': {
                'avg': sum(cpu_deltas) / len(cpu_deltas),
                'min': min(cpu_deltas),
                'max': max(cpu_deltas)
            },
            'memory_peak_mb': {
                'avg': sum(op.memory_peak_mb for op in ops) / len(ops),
                'min': min(op.memory_peak_mb for op in ops),
                'max': max(op.memory_peak_mb for op in ops)
            }
        }
        
    def get_operation_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all unique operations."""
        with self.lock:
            unique_operations = set(op.operation_name for op in self.operations)
            
        summary = {}
        for op_name in unique_operations:
            summary[op_name] = self.get_operation_stats(op_name)
            
        return summary
        
    def save_data(self) -> bool:
        """Save performance data to JSON file."""
        if not self.operations:
            return False
            
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'total_operations': len(self.operations),
                'operations': [asdict(op) for op in self.operations],
                'summary': self.get_operation_summary()
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
            return False
            
    def clear_data(self):
        """Clear all collected performance data."""
        with self.lock:
            self.operations.clear()
            self.operation_stack.clear()
            
    def get_slowest_operations(self, limit: int = 10) -> List[OperationMetrics]:
        """Get the slowest operations by duration."""
        with self.lock:
            return sorted(self.operations, key=lambda x: x.duration_ms, reverse=True)[:limit]
            
    def get_memory_intensive_operations(self, limit: int = 10) -> List[OperationMetrics]:
        """Get operations with highest memory usage."""
        with self.lock:
            return sorted(self.operations, 
                         key=lambda x: x.memory_peak_mb - x.memory_before_mb, 
                         reverse=True)[:limit]
                        
    def enable(self):
        """Enable performance profiling."""
        self.enabled = True
        
    def disable(self):
        """Disable performance profiling."""
        self.enabled = False
        
    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled
        
    def toggle(self):
        """Toggle profiling on/off."""
        self.enabled = not self.enabled
        return self.enabled


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> Optional[PerformanceProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def init_profiler(enabled: bool = True, log_file: str = "performance_profile.json") -> PerformanceProfiler:
    """Initialize the global profiler."""
    global _global_profiler
    _global_profiler = PerformanceProfiler(enabled, log_file)
    return _global_profiler


def profile_operation(operation_name: str):
    """
    Decorator for profiling functions.
    
    Usage:
        @profile_operation("my_function")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            if profiler and profiler.enabled:
                with profiler.profile_operation(operation_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_method(operation_name: Optional[str] = None):
    """
    Decorator for profiling methods.
    
    Usage:
        class MyClass:
            @profile_method()
            def my_method(self):
                pass
                
            @profile_method("custom_name")
            def another_method(self):
                pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            profiler = get_profiler()
            if profiler and profiler.enabled:
                name = operation_name or f"{self.__class__.__name__}.{func.__name__}"
                with profiler.profile_operation(name):
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator
