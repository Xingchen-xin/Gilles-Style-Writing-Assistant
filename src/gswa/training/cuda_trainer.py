"""
CUDA/PyTorch Training Integration - Linux/Windows LoRA Fine-tuning.

Integrates the GSWA training infrastructure with PyTorch/PEFT for NVIDIA GPUs:
- Hardware-aware auto-configuration
- OOM-aware fallback strategy
- Data preprocessing for long sequences
- Structured logging and visualization
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from .hardware import HardwareDetector, HardwareInfo
from .run_manager import RunManager, RunConfig
from .preprocessor import DataPreprocessor
from .logger import TrainingLogger


@dataclass
class CUDATrainingConfig:
    """Configuration for CUDA/PyTorch LoRA training."""
    # Model settings
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    quantize: str = "4bit"  # none, 4bit, 8bit

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training settings
    batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Data settings
    training_data: str = "./data/training/alpaca_train.jsonl"
    preprocess_enabled: bool = True

    # Output settings
    output_dir: str = "./models"
    run_name: Optional[str] = None

    # Safety settings
    enable_oom_fallback: bool = True
    max_fallback_attempts: int = 4
    oom_margin: float = 0.8

    # Hardware
    use_cpu: bool = False
    device_map: str = "auto"

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CUDATrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CUDAFallbackState:
    """Tracks OOM fallback state for CUDA training."""
    original_config: Dict[str, Any] = field(default_factory=dict)
    current_attempt: int = 0
    fallback_history: List[Dict[str, Any]] = field(default_factory=list)

    # Fallback order for CUDA (different from MLX)
    FALLBACK_ORDER = [
        ("eval_batch_size", 0.5),    # Halve eval batch
        ("batch_size", 0.5),          # Halve train batch
        ("max_seq_length", 0.75),     # Reduce seq length
        ("gradient_accumulation_steps", 2.0),  # Double grad accum
    ]


class CUDATrainer:
    """Manages CUDA/PyTorch LoRA training with GSWA infrastructure."""

    def __init__(self, config: CUDATrainingConfig):
        self.config = config
        self.hardware = HardwareDetector.detect()
        self.run_manager: Optional[RunManager] = None
        self.logger: Optional[TrainingLogger] = None
        self.fallback_state = CUDAFallbackState()
        self.fallback_state.original_config = config.to_dict()

    def prepare_data(self) -> Optional[str]:
        """Preprocess training data if needed."""
        if not self.config.preprocess_enabled:
            return self.config.training_data

        data_path = Path(self.config.training_data)
        if not data_path.exists():
            print(f"ERROR: Training data not found: {data_path}")
            return None

        # Create preprocessor
        preprocessor = DataPreprocessor(
            max_tokens=self.config.max_seq_length,
            model_id=self.config.model_id,
        )

        # Analyze first
        stats = preprocessor.analyze_data(str(data_path))

        print("\n" + "=" * 60)
        print("Data Analysis")
        print("=" * 60)
        print(f"  Total entries: {stats.total_entries}")
        print(f"  Token distribution: min={stats.min_tokens}, "
              f"median={stats.median_tokens}, p95={stats.p95_tokens}, max={stats.max_tokens}")
        print(f"  Truncation at {self.config.max_seq_length}: {stats.truncation_percentage:.1f}%")

        # Preprocess if needed
        if stats.truncation_percentage > 5:
            print(f"\n  Preprocessing to eliminate truncation...")
            output_path = data_path.parent / f"preprocessed_{data_path.name}"
            result = preprocessor.preprocess(str(data_path), str(output_path))

            print(f"  Original entries: {result['original_entries']}")
            print(f"  Output entries: {result['output_entries']}")
            print(f"  New truncation: {result['new_truncation_pct']:.1f}%")

            return str(output_path)
        else:
            print(f"  Data is already well-sized, no preprocessing needed.")
            return self.config.training_data

    def auto_configure(self) -> CUDATrainingConfig:
        """Auto-configure training parameters based on hardware."""
        vram = self.hardware.gpu_memory_gb or 8.0

        print("\n" + "=" * 60)
        print("Auto-Configuration")
        print("=" * 60)
        print(f"  GPU: {self.hardware.gpu_name or 'Unknown'}")
        print(f"  VRAM: {vram:.1f} GB")

        # Select profile based on VRAM
        if vram >= 48:
            profile = "high_end"
            self.config.batch_size = 8
            self.config.eval_batch_size = 8
            self.config.gradient_accumulation_steps = 1
            self.config.max_seq_length = 2048
            self.config.quantize = "none"
            self.config.lora_r = 32
            self.config.lora_alpha = 64
        elif vram >= 24:
            profile = "professional"
            self.config.batch_size = 4
            self.config.eval_batch_size = 4
            self.config.gradient_accumulation_steps = 2
            self.config.max_seq_length = 2048
            self.config.quantize = "4bit"
            self.config.lora_r = 16
            self.config.lora_alpha = 32
        elif vram >= 16:
            profile = "standard"
            self.config.batch_size = 2
            self.config.eval_batch_size = 2
            self.config.gradient_accumulation_steps = 4
            self.config.max_seq_length = 1536
            self.config.quantize = "4bit"
            self.config.lora_r = 16
            self.config.lora_alpha = 32
        elif vram >= 8:
            profile = "entry"
            self.config.batch_size = 1
            self.config.eval_batch_size = 1
            self.config.gradient_accumulation_steps = 8
            self.config.max_seq_length = 1024
            self.config.quantize = "4bit"
            self.config.lora_r = 8
            self.config.lora_alpha = 16
        else:
            profile = "minimal"
            self.config.batch_size = 1
            self.config.eval_batch_size = 1
            self.config.gradient_accumulation_steps = 16
            self.config.max_seq_length = 512
            self.config.quantize = "4bit"
            self.config.lora_r = 4
            self.config.lora_alpha = 8

        print(f"  Profile: {profile}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Sequence length: {self.config.max_seq_length}")
        print(f"  Quantization: {self.config.quantize}")
        print(f"  LoRA rank: {self.config.lora_r}")

        return self.config

    def apply_fallback(self) -> bool:
        """Apply next fallback strategy after OOM."""
        if self.fallback_state.current_attempt >= self.config.max_fallback_attempts:
            return False

        fallback_idx = self.fallback_state.current_attempt
        if fallback_idx >= len(CUDAFallbackState.FALLBACK_ORDER):
            return False

        param_name, factor = CUDAFallbackState.FALLBACK_ORDER[fallback_idx]
        old_value = getattr(self.config, param_name)

        if factor < 1:
            new_value = max(1, int(old_value * factor))
        else:
            new_value = int(old_value * factor)

        setattr(self.config, param_name, new_value)

        fallback_info = {
            "attempt": self.fallback_state.current_attempt + 1,
            "parameter": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.now().isoformat(),
        }

        self.fallback_state.fallback_history.append(fallback_info)
        self.fallback_state.current_attempt += 1

        print(f"\n[OOM FALLBACK] Attempt {fallback_info['attempt']}: "
              f"{param_name} {old_value} -> {new_value}")

        if self.logger:
            self.logger.log_event("oom_fallback", fallback_info)

        return True

    def train(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run training with full infrastructure integration."""
        # Initialize run manager
        run_name = self.config.run_name or f"cuda-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.run_manager = RunManager(
            base_dir=self.config.output_dir,
            run_name=run_name,
        )

        run_dir = self.run_manager.create_run()

        # Save initial config
        config_dir = run_dir / "config"
        config_dir.mkdir(exist_ok=True)

        with open(config_dir / "run_config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        with open(config_dir / "hardware_info.json", 'w') as f:
            json.dump(self.hardware.to_dict(), f, indent=2)

        # Initialize logger
        log_dir = run_dir / "logs"
        self.logger = TrainingLogger(str(log_dir))

        # Prepare data
        training_data = self.prepare_data()
        if not training_data:
            return {"success": False, "error": "Data preparation failed"}

        self.config.training_data = training_data

        # Training loop with OOM fallback
        while True:
            try:
                result = self._run_training_subprocess(run_dir, progress_callback)

                if result.get("success"):
                    # Generate visualizations
                    self._generate_visualizations(run_dir)

                    # Update metadata
                    self.run_manager.update_metadata({
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "final_loss": result.get("final_loss"),
                    })

                    return result

                elif result.get("error_type") == "oom":
                    if self.apply_fallback():
                        print("\nRetrying with reduced parameters...")
                        continue
                    else:
                        return {
                            "success": False,
                            "error": "Max fallback attempts reached",
                            "fallback_history": self.fallback_state.fallback_history,
                        }
                else:
                    return result

            except KeyboardInterrupt:
                self.run_manager.update_metadata({
                    "status": "interrupted",
                    "interrupted_at": datetime.now().isoformat(),
                })
                return {"success": False, "error": "Interrupted by user"}

            except Exception as e:
                self.run_manager.update_metadata({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                })
                return {"success": False, "error": str(e)}

    def _run_training_subprocess(
        self,
        run_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run the actual training as subprocess for better OOM handling."""
        # Build command for finetune_lora.py
        script_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        script_path = script_dir / "finetune_lora.py"

        cmd = [
            sys.executable, str(script_path),
            "--base-model", self.config.model_id,
            "--training-data", self.config.training_data,
            "--output-dir", str(run_dir / "adapters"),
            "--quantize", self.config.quantize,
            "--lora-r", str(self.config.lora_r),
            "--lora-alpha", str(self.config.lora_alpha),
            "--lora-dropout", str(self.config.lora_dropout),
            "--epochs", str(self.config.epochs),
            "--batch-size", str(self.config.batch_size),
            "--gradient-accumulation-steps", str(self.config.gradient_accumulation_steps),
            "--learning-rate", str(self.config.learning_rate),
            "--max-length", str(self.config.max_seq_length),
        ]

        if self.config.use_cpu:
            cmd.append("--cpu")

        print("\n" + "=" * 60)
        print("Starting CUDA Training")
        print("=" * 60)
        print(f"  Model: {self.config.model_id}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Sequence length: {self.config.max_seq_length}")
        print(f"  Quantization: {self.config.quantize}")
        print(f"  Output: {run_dir}")

        # Run training
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Let output go to terminal
                text=True,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "output_dir": str(run_dir),
                }
            else:
                # Check if OOM
                # Note: PyTorch OOM typically exits with non-zero
                return {
                    "success": False,
                    "error_type": "training_failed",
                    "returncode": result.returncode,
                }

        except subprocess.CalledProcessError as e:
            if "CUDA out of memory" in str(e.stderr or ""):
                return {
                    "success": False,
                    "error_type": "oom",
                }
            raise

    def _generate_visualizations(self, run_dir: Path):
        """Generate visualizations from training logs."""
        try:
            from .visualizer import TrainingVisualizer, generate_training_report

            log_dir = run_dir / "logs"
            report_dir = run_dir / "reports"
            report_dir.mkdir(exist_ok=True)

            # Generate report
            config = self.config.to_dict()
            report_path = generate_training_report(
                str(log_dir),
                str(report_dir),
                config
            )

            print(f"\nReport generated: {report_path}")

        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")


def run_cuda_training(
    model_id: Optional[str] = None,
    training_data: Optional[str] = None,
    output_dir: str = "./models",
    auto_config: bool = True,
    preprocess: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for one-click CUDA training.

    Args:
        model_id: Model to fine-tune (auto-detected if None)
        training_data: Path to training data
        output_dir: Output directory
        auto_config: Auto-configure based on hardware
        preprocess: Enable data preprocessing
        **kwargs: Additional config overrides

    Returns:
        Training result dict
    """
    # Create config
    config = CUDATrainingConfig(
        model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3",
        training_data=training_data or "./data/training/alpaca_train.jsonl",
        output_dir=output_dir,
        preprocess_enabled=preprocess,
    )

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create trainer
    trainer = CUDATrainer(config)

    # Auto-configure if requested
    if auto_config:
        trainer.auto_configure()

    # Run training
    return trainer.train()


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="CUDA Training Integration Test")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--data", default="./data/training/alpaca_train.jsonl")
    parser.add_argument("--output", default="./models")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.dry_run:
        config = CUDATrainingConfig(model_id=args.model)
        trainer = CUDATrainer(config)
        trainer.auto_configure()
        print("\nDry run complete. Would train with:")
        print(json.dumps(trainer.config.to_dict(), indent=2))
    else:
        result = run_cuda_training(
            model_id=args.model,
            training_data=args.data,
            output_dir=args.output,
        )
        print("\nResult:", json.dumps(result, indent=2))
