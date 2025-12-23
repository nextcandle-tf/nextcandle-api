"""
Performance optimization with caching and async processing
"""
import functools
import time
import threading
from typing import Dict, Any, Optional, Callable
import json
import hashlib

class PerformanceCache:
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.ttl = ttl
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self._access_times:
            return True
        return time.time() - self._access_times[key] > self.ttl
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self._access_times.items()
            if current_time - access_time > self.ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def _evict_lru(self):
        """Remove least recently used entries if cache is full"""
        if len(self._cache) <= self.max_size:
            return
        
        # Sort by access time and remove oldest
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
        keys_to_remove = sorted_keys[:len(self._cache) - self.max_size + 1]
        
        for key, _ in keys_to_remove:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def get(self, func_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached result"""
        with self._lock:
            key = self._generate_key(func_name, args, kwargs)
            
            if key not in self._cache or self._is_expired(key):
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return self._cache[key]['result']
    
    def set(self, func_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache result"""
        with self._lock:
            key = self._generate_key(func_name, args, kwargs)
            current_time = time.time()
            
            self._cache[key] = {
                'result': result,
                'cached_at': current_time
            }
            self._access_times[key] = current_time
            
            # Cleanup
            self._evict_expired()
            self._evict_lru()
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'keys': list(self._cache.keys())
            }

# Global cache instance
performance_cache = PerformanceCache(max_size=200, ttl=600)  # 10분 TTL

def cached(ttl: int = 300):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache first
            cached_result = performance_cache.get(func.__name__, args, kwargs)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            performance_cache.set(func.__name__, args, kwargs, result)
            return result
        
        return wrapper
    return decorator

class AsyncTaskManager:
    """Manage background tasks for better performance"""
    def __init__(self):
        self._tasks: Dict[str, threading.Thread] = {}
        self._results: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def start_task(self, task_id: str, target: Callable, args: tuple = (), kwargs: dict = None):
        """Start background task"""
        if kwargs is None:
            kwargs = {}
        
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id].is_alive():
                return False  # Task already running
            
            def task_wrapper():
                try:
                    result = target(*args, **kwargs)
                    with self._lock:
                        self._results[task_id] = {'status': 'completed', 'result': result}
                except Exception as e:
                    with self._lock:
                        self._results[task_id] = {'status': 'error', 'error': str(e)}
            
            thread = threading.Thread(target=task_wrapper, daemon=True)
            self._tasks[task_id] = thread
            self._results[task_id] = {'status': 'running'}
            thread.start()
            return True
    
    def get_result(self, task_id: str) -> Optional[dict]:
        """Get task result"""
        with self._lock:
            return self._results.get(task_id)
    
    def is_running(self, task_id: str) -> bool:
        """Check if task is running"""
        with self._lock:
            return (task_id in self._tasks and 
                   self._tasks[task_id].is_alive() and
                   self._results.get(task_id, {}).get('status') == 'running')
    
    def cleanup_completed(self):
        """Clean up completed tasks"""
        with self._lock:
            completed_tasks = [
                task_id for task_id, thread in self._tasks.items()
                if not thread.is_alive()
            ]
            for task_id in completed_tasks:
                self._tasks.pop(task_id, None)

# Global task manager
task_manager = AsyncTaskManager()