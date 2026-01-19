"""
GSWA Training Module - MLX LoRA Fine-tuning Infrastructure.

This module provides:
- Hardware-aware auto-configuration
- OOM-aware fallback strategy
- Data preprocessing for long sequences
- Training plan selection with dry-run
- Structured logging and visualization
"""

from .hardware import HardwareDetector, HardwareInfo
from .run_manager import RunManager, RunConfig
from .planner import TrainingPlanner, PlanCandidate
from .logger import TrainingLogger
from .preprocessor import DataPreprocessor, PreprocessStats

__all__ = [
    "HardwareDetector",
    "HardwareInfo",
    "RunManager",
    "RunConfig",
    "TrainingPlanner",
    "PlanCandidate",
    "TrainingLogger",
    "DataPreprocessor",
    "PreprocessStats",
]
