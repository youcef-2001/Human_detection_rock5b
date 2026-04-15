"""Platform detection utilities for selecting appropriate inference backend."""

import platform
import os
from enum import Enum


class InferencePlatform(Enum):
    """Supported inference platforms."""
    
    NPU_RK3588 = "npu_rk3588"
    CPU_ONNX = "cpu_onnx"


class PlatformDetector:
    """
    Detect the current hardware platform and determine optimal inference backend.
    
    Attributes:
        _instance: Singleton instance for platform detection.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern to avoid redundant detection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._detected_platform = None
        return cls._instance
    
    @property
    def platform(self) -> InferencePlatform:
        """
        Lazy-load and cache the detected platform.
        
        Returns:
            InferencePlatform: The detected inference platform.
        """
        if self._detected_platform is None:
            self._detected_platform = self._detect_platform()
        return self._detected_platform
    
    @staticmethod
    def _detect_platform() -> InferencePlatform:
        """
        Detect the current hardware platform by checking system characteristics.
        
        Returns:
            InferencePlatform: Detected platform (NPU_RK3588 or CPU_ONNX).
        """
        machine = platform.machine()
        system = platform.system()
        
        # Check for ARM64 architecture (RK3588 typically uses ARM64)
        if machine in ("aarch64", "arm64"):
            # Check for RK3588/RK3566 specific markers
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpu_info = f.read().lower()
                    if "rockchip" in cpu_info or "rk3588" in cpu_info or "rk3566" in cpu_info:
                        return InferencePlatform.NPU_RK3588
            except (FileNotFoundError, PermissionError):
                pass
        
        # Default to CPU ONNX for x86/x64 or unknown platforms
        return InferencePlatform.CPU_ONNX
    
    @staticmethod
    def is_npu_available() -> bool:
        """
        Check if NPU (RKNN) is available on this platform.
        
        Returns:
            bool: True if running on RK3588, False otherwise.
        """
        try:
            from rknnlite.api import RKNNLite  # noqa: F401
            detector = PlatformDetector()
            return detector.platform == InferencePlatform.NPU_RK3588
        except ImportError:
            return False
    
    @staticmethod
    def is_onnx_available() -> bool:
        """
        Check if ONNX Runtime is available.
        
        Returns:
            bool: True if onnxruntime can be imported, False otherwise.
        """
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False
