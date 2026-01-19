"""
GSWA Training Module - MLX LoRA Fine-tuning Infrastructure.

This module provides:
- Hardware-aware auto-configuration
- OOM-aware fallback strategy
- Data preprocessing for long sequences
- Training plan selection with dry-run
- Structured logging and visualization
- Real-time training metrics parsing
"""

from .hardware import HardwareDetector, HardwareInfo
from .run_manager import RunManager, RunConfig
from .planner import TrainingPlanner, PlanCandidate
from .logger import TrainingLogger
from .preprocessor import DataPreprocessor, PreprocessStats
from .metrics_parser import MLXMetricsParser, create_ascii_loss_graph
from .visualizer import TrainingVisualizer, generate_training_report

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
    "MLXMetricsParser",
    "create_ascii_loss_graph",
    "TrainingVisualizer",
    "generate_training_report",
]
