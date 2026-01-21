"""
Hardware Detection Module - Detects Apple Silicon and Memory.

Provides detailed hardware information for auto-configuration:
- Apple Silicon chip type (M1, M2, M3, etc.)
- Total and available unified memory
- GPU cores and neural engine
- Safe training parameters based on hardware
"""

import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import os


@dataclass
class HardwareInfo:
    """Detailed hardware information."""
    # System
    os_name: str = ""
    os_version: str = ""
    hostname: str = ""

    # CPU/Chip
    chip_name: str = ""
    chip_type: str = ""  # "apple_silicon", "intel", "other"
    cpu_cores: int = 0
    performance_cores: int = 0
    efficiency_cores: int = 0

    # Memory
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    memory_pressure: str = "normal"  # "normal", "warn", "critical"

    # GPU (Apple Silicon unified memory)
    gpu_cores: int = 0
    gpu_memory_gb: float = 0.0  # For Apple Silicon, this is derived from unified memory
    neural_engine_cores: int = 0

    # MLX/Metal availability
    mlx_available: bool = False
    mlx_version: str = ""
    metal_available: bool = False
    metal_version: str = ""

    # CUDA (for non-Mac)
    cuda_available: bool = False
    cuda_version: str = ""
    cuda_device_name: str = ""
    cuda_device_memory_gb: float = 0.0

    # Recommended settings
    recommended_backend: str = "cpu"  # "mlx", "cuda", "cpu"
    recommended_batch_size: int = 1
    recommended_max_seq_length: int = 512
    recommended_num_layers: int = 4
    recommended_eval_batch_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareInfo":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class HardwareDetector:
    """Detects hardware capabilities for training optimization."""

    # Memory thresholds for Apple Silicon unified memory
    # Conservative estimates to avoid OOM
    MEMORY_CONFIGS = {
        # (min_gb, max_gb): (batch_size, num_layers, max_seq_length, eval_batch_size)
        (0, 8): (1, 4, 512, 1),
        (8, 16): (1, 8, 768, 1),
        (16, 24): (1, 8, 1024, 1),
        (24, 36): (2, 12, 1536, 1),
        (36, 48): (2, 16, 1536, 2),
        (48, 64): (4, 16, 2048, 2),
        (64, 96): (4, 24, 2048, 2),
        (96, 128): (8, 24, 2048, 4),
        (128, 256): (8, 32, 4096, 4),
    }

    # Apple Silicon chip configurations
    APPLE_CHIPS = {
        "M1": {"gpu_cores": 8, "neural_cores": 16, "perf_cores": 4, "eff_cores": 4},
        "M1 Pro": {"gpu_cores": 16, "neural_cores": 16, "perf_cores": 8, "eff_cores": 2},
        "M1 Max": {"gpu_cores": 32, "neural_cores": 16, "perf_cores": 8, "eff_cores": 2},
        "M1 Ultra": {"gpu_cores": 64, "neural_cores": 32, "perf_cores": 16, "eff_cores": 4},
        "M2": {"gpu_cores": 10, "neural_cores": 16, "perf_cores": 4, "eff_cores": 4},
        "M2 Pro": {"gpu_cores": 19, "neural_cores": 16, "perf_cores": 8, "eff_cores": 4},
        "M2 Max": {"gpu_cores": 38, "neural_cores": 16, "perf_cores": 8, "eff_cores": 4},
        "M2 Ultra": {"gpu_cores": 76, "neural_cores": 32, "perf_cores": 16, "eff_cores": 8},
        "M3": {"gpu_cores": 10, "neural_cores": 16, "perf_cores": 4, "eff_cores": 4},
        "M3 Pro": {"gpu_cores": 18, "neural_cores": 16, "perf_cores": 6, "eff_cores": 6},
        "M3 Max": {"gpu_cores": 40, "neural_cores": 16, "perf_cores": 12, "eff_cores": 4},
        "M4": {"gpu_cores": 10, "neural_cores": 16, "perf_cores": 4, "eff_cores": 6},
        "M4 Pro": {"gpu_cores": 20, "neural_cores": 16, "perf_cores": 10, "eff_cores": 4},
        "M4 Max": {"gpu_cores": 40, "neural_cores": 16, "perf_cores": 14, "eff_cores": 4},
    }

    def __init__(self):
        self.info = HardwareInfo()

    def detect(self) -> HardwareInfo:
        """Detect all hardware information."""
        self._detect_system()
        self._detect_cpu()
        self._detect_memory()
        self._detect_gpu()
        self._detect_frameworks()
        self._calculate_recommendations()
        return self.info

    def _detect_system(self):
        """Detect basic system information."""
        self.info.os_name = platform.system()
        self.info.os_version = platform.version()
        self.info.hostname = platform.node()

    def _detect_cpu(self):
        """Detect CPU/chip information."""
        self.info.cpu_cores = os.cpu_count() or 1

        if self.info.os_name == "Darwin":
            self._detect_apple_silicon()
        elif self.info.os_name == "Linux":
            self._detect_linux_cpu()
        elif self.info.os_name == "Windows":
            self._detect_windows_cpu()

    def _detect_apple_silicon(self):
        """Detect Apple Silicon chip details."""
        try:
            # Get chip name
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            chip_name = result.stdout.strip()
            self.info.chip_name = chip_name

            if "Apple" in chip_name:
                self.info.chip_type = "apple_silicon"

                # Parse chip variant
                for chip_key in sorted(self.APPLE_CHIPS.keys(), key=len, reverse=True):
                    if chip_key in chip_name:
                        chip_info = self.APPLE_CHIPS[chip_key]
                        self.info.gpu_cores = chip_info["gpu_cores"]
                        self.info.neural_engine_cores = chip_info["neural_cores"]
                        self.info.performance_cores = chip_info["perf_cores"]
                        self.info.efficiency_cores = chip_info["eff_cores"]
                        break
            else:
                self.info.chip_type = "intel"

        except Exception:
            self.info.chip_type = "unknown"

    def _detect_linux_cpu(self):
        """Detect Linux CPU info."""
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        self.info.chip_name = line.split(":")[1].strip()
                        break
            self.info.chip_type = "other"
        except Exception:
            self.info.chip_type = "unknown"

    def _detect_windows_cpu(self):
        """Detect Windows CPU info."""
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, shell=True
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                self.info.chip_name = lines[1].strip()
            self.info.chip_type = "other"
        except Exception:
            self.info.chip_type = "unknown"

    def _detect_memory(self):
        """Detect memory information."""
        if self.info.os_name == "Darwin":
            self._detect_mac_memory()
        elif self.info.os_name == "Linux":
            self._detect_linux_memory()
        elif self.info.os_name == "Windows":
            self._detect_windows_memory()

        # Calculate GPU memory for Apple Silicon (unified memory)
        if self.info.chip_type == "apple_silicon":
            # GPU can use up to ~75% of unified memory for training
            self.info.gpu_memory_gb = self.info.total_memory_gb * 0.75

    def _detect_mac_memory(self):
        """Detect macOS memory."""
        try:
            # Total memory
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            self.info.total_memory_gb = int(result.stdout.strip()) / (1024 ** 3)

            # Memory pressure via vm_stat
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True, text=True
            )
            lines = result.stdout.strip().split("\n")

            page_size = 4096  # Default
            free_pages = 0
            inactive_pages = 0

            for line in lines:
                if "page size" in line.lower():
                    try:
                        page_size = int(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
                elif "Pages free" in line:
                    try:
                        free_pages = int(line.split()[-1].rstrip("."))
                    except (ValueError, IndexError):
                        pass
                elif "Pages inactive" in line:
                    try:
                        inactive_pages = int(line.split()[-1].rstrip("."))
                    except (ValueError, IndexError):
                        pass

            self.info.available_memory_gb = (free_pages + inactive_pages) * page_size / (1024 ** 3)

            # Determine memory pressure
            used_ratio = 1 - (self.info.available_memory_gb / self.info.total_memory_gb)
            if used_ratio > 0.9:
                self.info.memory_pressure = "critical"
            elif used_ratio > 0.75:
                self.info.memory_pressure = "warn"
            else:
                self.info.memory_pressure = "normal"

        except Exception:
            self.info.total_memory_gb = 16.0  # Default assumption
            self.info.available_memory_gb = 8.0

    def _detect_linux_memory(self):
        """Detect Linux memory."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        self.info.total_memory_gb = int(line.split()[1]) / (1024 ** 2)
                    elif line.startswith("MemAvailable:"):
                        self.info.available_memory_gb = int(line.split()[1]) / (1024 ** 2)
        except Exception:
            self.info.total_memory_gb = 16.0
            self.info.available_memory_gb = 8.0

    def _detect_windows_memory(self):
        """Detect Windows memory."""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

            self.info.total_memory_gb = stat.ullTotalPhys / (1024 ** 3)
            self.info.available_memory_gb = stat.ullAvailPhys / (1024 ** 3)
        except Exception:
            self.info.total_memory_gb = 16.0
            self.info.available_memory_gb = 8.0

    def _detect_gpu(self):
        """Detect GPU information."""
        if self.info.chip_type == "apple_silicon":
            # Already handled in _detect_apple_silicon and _detect_memory
            pass
        else:
            self._detect_nvidia_gpu()

    def _detect_nvidia_gpu(self):
        """Detect NVIDIA GPU. Supports multi-GPU setups by summing VRAM."""
        gpu_count = 0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpu_count = len(lines)
                total_vram_mb = 0
                gpu_names = []

                for line in lines:
                    parts = line.split(",")
                    name = parts[0].strip()
                    vram_mb = float(parts[1].strip())
                    gpu_names.append(name)
                    total_vram_mb += vram_mb

                # Display name: show count if multiple GPUs
                if gpu_count > 1:
                    self.info.cuda_device_name = f"{gpu_names[0]} x{gpu_count}"
                else:
                    self.info.cuda_device_name = gpu_names[0]

                self.info.cuda_device_memory_gb = total_vram_mb / 1024  # Total VRAM across all GPUs
        except Exception:
            pass

        # Check CUDA via PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                self.info.cuda_available = True
                self.info.cuda_version = torch.version.cuda or ""
                if not self.info.cuda_device_name:
                    gpu_count = torch.cuda.device_count()
                    total_vram = sum(
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(gpu_count)
                    )
                    self.info.cuda_device_name = torch.cuda.get_device_name(0)
                    if gpu_count > 1:
                        self.info.cuda_device_name = f"{self.info.cuda_device_name} x{gpu_count}"
                    self.info.cuda_device_memory_gb = total_vram / (1024 ** 3)
        except ImportError:
            pass

    def _detect_frameworks(self):
        """Detect available ML frameworks."""
        # Check MLX
        try:
            import mlx
            import mlx.core as mx
            self.info.mlx_available = True
            try:
                from importlib.metadata import version
                self.info.mlx_version = version('mlx')
            except Exception:
                self.info.mlx_version = "unknown"

            # Check Metal through MLX
            try:
                device = str(mx.default_device())
                if "gpu" in device.lower():
                    self.info.metal_available = True
            except Exception:
                pass
        except ImportError:
            pass

    def _calculate_recommendations(self):
        """Calculate recommended training parameters."""
        # Determine backend
        if self.info.mlx_available and self.info.chip_type == "apple_silicon":
            self.info.recommended_backend = "mlx"
            mem_gb = self.info.total_memory_gb
        elif self.info.cuda_available:
            self.info.recommended_backend = "cuda"
            mem_gb = self.info.cuda_device_memory_gb
        else:
            self.info.recommended_backend = "cpu"
            mem_gb = self.info.total_memory_gb * 0.5

        # Apply memory pressure adjustment
        if self.info.memory_pressure == "critical":
            mem_gb *= 0.5
        elif self.info.memory_pressure == "warn":
            mem_gb *= 0.7

        # Find appropriate configuration
        for (min_gb, max_gb), (batch, layers, seq_len, eval_batch) in self.MEMORY_CONFIGS.items():
            if min_gb <= mem_gb < max_gb:
                self.info.recommended_batch_size = batch
                self.info.recommended_num_layers = layers
                self.info.recommended_max_seq_length = seq_len
                self.info.recommended_eval_batch_size = eval_batch
                break
        else:
            # Default to most conservative
            self.info.recommended_batch_size = 1
            self.info.recommended_num_layers = 4
            self.info.recommended_max_seq_length = 512
            self.info.recommended_eval_batch_size = 1

    def get_safe_config(self, margin: float = 0.8) -> Dict[str, Any]:
        """Get safe training configuration with safety margin.

        Args:
            margin: Safety margin (0.8 = use 80% of estimated safe values)
        """
        config = {
            "batch_size": max(1, int(self.info.recommended_batch_size * margin)),
            "num_layers": max(4, int(self.info.recommended_num_layers * margin)),
            "max_seq_length": max(256, int(self.info.recommended_max_seq_length * margin / 256) * 256),
            "eval_batch_size": max(1, int(self.info.recommended_eval_batch_size * margin)),
            "backend": self.info.recommended_backend,
        }
        return config

    def print_summary(self):
        """Print hardware summary."""
        print("\n" + "=" * 60)
        print("Hardware Detection Summary")
        print("=" * 60)
        print(f"\nSystem: {self.info.os_name} ({self.info.os_version[:40]}...)")
        print(f"Chip: {self.info.chip_name}")
        print(f"Type: {self.info.chip_type}")
        print(f"\nMemory:")
        print(f"  Total: {self.info.total_memory_gb:.1f} GB")
        print(f"  Available: {self.info.available_memory_gb:.1f} GB")
        print(f"  Pressure: {self.info.memory_pressure}")

        if self.info.chip_type == "apple_silicon":
            print(f"\nApple Silicon:")
            print(f"  GPU Cores: {self.info.gpu_cores}")
            print(f"  Neural Engine: {self.info.neural_engine_cores} cores")
            print(f"  GPU Memory (unified): {self.info.gpu_memory_gb:.1f} GB")

        if self.info.cuda_available:
            print(f"\nNVIDIA GPU:")
            print(f"  Device: {self.info.cuda_device_name}")
            print(f"  VRAM: {self.info.cuda_device_memory_gb:.1f} GB")
            print(f"  CUDA: {self.info.cuda_version}")

        print(f"\nFrameworks:")
        print(f"  MLX: {'Yes (' + self.info.mlx_version + ')' if self.info.mlx_available else 'No'}")
        print(f"  Metal: {'Yes' if self.info.metal_available else 'No'}")
        print(f"  CUDA: {'Yes (' + self.info.cuda_version + ')' if self.info.cuda_available else 'No'}")

        print(f"\nRecommended Settings:")
        print(f"  Backend: {self.info.recommended_backend}")
        print(f"  batch_size: {self.info.recommended_batch_size}")
        print(f"  max_seq_length: {self.info.recommended_max_seq_length}")
        print(f"  num_layers: {self.info.recommended_num_layers}")
        print(f"  eval_batch_size: {self.info.recommended_eval_batch_size}")


def detect_hardware() -> HardwareInfo:
    """Convenience function to detect hardware."""
    detector = HardwareDetector()
    return detector.detect()


if __name__ == "__main__":
    detector = HardwareDetector()
    info = detector.detect()
    detector.print_summary()
    print("\nJSON output:")
    print(info.to_json())
