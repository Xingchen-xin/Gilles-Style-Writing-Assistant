"""
GSWA Training Module - Cross-Platform LoRA Fine-tuning Infrastructure.

This module provides:
- Hardware-aware auto-configuration (Apple Silicon + NVIDIA CUDA)
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
from .cuda_trainer import CUDATrainer, CUDATrainingConfig, run_cuda_training

__all__ = [
    # Hardware detection
    "HardwareDetector",
    "HardwareInfo",
    # Run management
    "RunManager",
    "RunConfig",
    # Training planning
    "TrainingPlanner",
    "PlanCandidate",
    # Logging
    "TrainingLogger",
    # Data preprocessing
    "DataPreprocessor",
    "PreprocessStats",
    # Metrics parsing
    "MLXMetricsParser",
    "create_ascii_loss_graph",
    # Visualization
    "TrainingVisualizer",
    "generate_training_report",
    # CUDA/PyTorch training
    "CUDATrainer",
    "CUDATrainingConfig",
    "run_cuda_training",
]
