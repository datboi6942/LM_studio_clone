"""GPU monitoring service for real-time telemetry."""

import threading
import time
from typing import Any, Dict, Optional

import structlog

try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from flask_socketio import SocketIO

from ..config import Config

logger = structlog.get_logger(__name__)


class GPUMonitor:
    """Monitor GPU metrics and emit via SocketIO."""
    
    def __init__(self, socketio: SocketIO) -> None:
        """Initialize GPU monitor.
        
        Args:
            socketio: SocketIO instance for emitting metrics.
        """
        self.socketio = socketio
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.initialized = False
        
        if HAS_GPU:
            try:
                pynvml.nvmlInit()
                self.initialized = True
                logger.info("NVIDIA GPU monitoring initialized")
            except Exception as e:
                logger.warning("Failed to initialize GPU monitoring", error=str(e))
    
    def start(self) -> None:
        """Start monitoring in background thread."""
        if not self.initialized:
            logger.warning("GPU monitoring not available")
            return
        
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU monitoring started")
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.socketio.emit("gpu_metrics", metrics, namespace="/metrics")
            except Exception as e:
                logger.error("Error collecting GPU metrics", error=str(e))
            
            time.sleep(Config.GPU_MONITOR_INTERVAL)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics.
        
        Returns:
            Dictionary of GPU metrics.
        """
        metrics = {"gpus": []}
        
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get device info
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used // (1024 ** 2)
            memory_total_mb = mem_info.total // (1024 ** 2)
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory
            except:
                gpu_util = None
                mem_util = None
            
            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert to watts
            except:
                power = None
            
            metrics["gpus"].append({
                "index": i,
                "name": name,
                "memory": {
                    "used_mb": memory_used_mb,
                    "total_mb": memory_total_mb,
                    "percent": round(memory_percent, 1)
                },
                "temperature": temperature,
                "utilization": {
                    "gpu": gpu_util,
                    "memory": mem_util
                },
                "power_watts": power
            })
        
        metrics["timestamp"] = int(time.time())
        return metrics


# Global monitor instance
_monitor: Optional[GPUMonitor] = None


def start_gpu_monitoring(socketio: SocketIO) -> None:
    """Start GPU monitoring.
    
    Args:
        socketio: SocketIO instance.
    """
    global _monitor
    if not _monitor:
        _monitor = GPUMonitor(socketio)
        _monitor.start()


def stop_gpu_monitoring() -> None:
    """Stop GPU monitoring."""
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None 
