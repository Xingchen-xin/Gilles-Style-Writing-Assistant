"""
Run Manager Module - Manages training runs and configurations.

Provides:
- Timestamped run directories
- Configuration persistence (JSON/YAML)
- Run metadata tracking
- OOM fallback state management
"""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib


@dataclass
class RunConfig:
    """Training run configuration."""
    # Model
    model_id: str = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
    model_short: str = "mistral"

    # Training parameters
    batch_size: int = 1
    num_layers: int = 8
    max_seq_length: int = 1024
    learning_rate: float = 1e-5
    iters: int = 500
    grad_accum_steps: int = 1

    # Evaluation parameters
    eval_batch_size: int = 1
    eval_interval: int = 100

    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Data
    training_data: str = "./data/training/alpaca_train.jsonl"
    preprocessed_data: Optional[str] = None

    # OOM handling
    enable_oom_fallback: bool = True
    oom_fallback_count: int = 0
    original_config: Optional[Dict] = None

    # Preprocessing
    preprocess_enabled: bool = True
    chunk_overlap: int = 50

    # Seed for reproducibility
    seed: int = 42

    # Hardware info
    hardware_info: Optional[Dict] = None

    # Run metadata
    created_at: str = ""
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "RunConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "RunConfig":
        """Load from JSON/YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str):
        """Save to file (JSON or YAML based on extension)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
                except ImportError:
                    # Fallback to JSON
                    json.dump(self.to_dict(), f, indent=2)
            else:
                json.dump(self.to_dict(), f, indent=2)

    def get_config_hash(self) -> str:
        """Get hash of training-relevant config for reproducibility."""
        relevant_keys = [
            'model_id', 'batch_size', 'num_layers', 'max_seq_length',
            'learning_rate', 'iters', 'grad_accum_steps', 'lora_rank',
            'lora_alpha', 'lora_dropout', 'seed'
        ]
        relevant = {k: getattr(self, k) for k in relevant_keys}
        return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:8]


@dataclass
class OOMFallbackEvent:
    """Records an OOM fallback event."""
    timestamp: str
    error_message: str
    original_config: Dict[str, Any]
    new_config: Dict[str, Any]
    fallback_reason: str
    fallback_number: int


class RunManager:
    """Manages training run directories and configurations."""

    SUBDIRS = ["config", "logs", "plots", "adapters", "stats", "reports", "checkpoints"]

    def __init__(self, base_dir: str = "./runs"):
        self.base_dir = Path(base_dir)
        self.current_run_dir: Optional[Path] = None
        self.config: Optional[RunConfig] = None
        self.oom_events: List[OOMFallbackEvent] = []

    def create_run(
        self,
        config: RunConfig,
        run_name: Optional[str] = None,
        description: str = ""
    ) -> Path:
        """Create a new run directory with timestamp.

        Args:
            config: Training configuration
            run_name: Optional custom name (default: auto-generated)
            description: Optional description for the run

        Returns:
            Path to the run directory
        """
        # Generate run ID
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        config_hash = config.get_config_hash()

        if run_name:
            run_id = f"{timestamp}-{run_name}"
        else:
            run_id = f"{timestamp}-{config.model_short}-{config_hash}"

        # Create run directory
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in self.SUBDIRS:
            (run_dir / subdir).mkdir(exist_ok=True)

        # Update config with run metadata
        config.created_at = datetime.now().isoformat()
        config.run_id = run_id
        config.original_config = config.to_dict()

        # Save configuration
        config.save(run_dir / "config" / "run_config.json")

        # Save hardware info separately for reference
        if config.hardware_info:
            with open(run_dir / "config" / "hardware_info.json", 'w') as f:
                json.dump(config.hardware_info, f, indent=2)

        # Create run metadata
        metadata = {
            "run_id": run_id,
            "created_at": config.created_at,
            "description": description,
            "config_hash": config_hash,
            "status": "created",
            "model_id": config.model_id,
        }
        with open(run_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        self.current_run_dir = run_dir
        self.config = config

        return run_dir

    def resume_run(self, run_dir: str) -> RunConfig:
        """Resume a run from an existing directory.

        Args:
            run_dir: Path to the run directory

        Returns:
            Loaded RunConfig
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        config_path = run_dir / "config" / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        self.config = RunConfig.from_file(str(config_path))
        self.current_run_dir = run_dir

        # Load OOM events if any
        oom_log = run_dir / "logs" / "oom_events.jsonl"
        if oom_log.exists():
            self.oom_events = []
            with open(oom_log, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.oom_events.append(OOMFallbackEvent(**data))

        return self.config

    def update_status(self, status: str, message: str = ""):
        """Update run status in metadata."""
        if not self.current_run_dir:
            return

        metadata_path = self.current_run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata["status"] = status
        metadata["last_updated"] = datetime.now().isoformat()
        if message:
            metadata["status_message"] = message

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def record_oom_fallback(
        self,
        error_message: str,
        new_config: Dict[str, Any],
        reason: str
    ) -> OOMFallbackEvent:
        """Record an OOM fallback event.

        Args:
            error_message: The OOM error message
            new_config: The new configuration after fallback
            reason: Explanation of what was changed

        Returns:
            The created OOMFallbackEvent
        """
        if not self.config:
            raise RuntimeError("No active run")

        self.config.oom_fallback_count += 1

        event = OOMFallbackEvent(
            timestamp=datetime.now().isoformat(),
            error_message=error_message,
            original_config=self.config.to_dict(),
            new_config=new_config,
            fallback_reason=reason,
            fallback_number=self.config.oom_fallback_count,
        )
        self.oom_events.append(event)

        # Save to log
        if self.current_run_dir:
            oom_log = self.current_run_dir / "logs" / "oom_events.jsonl"
            with open(oom_log, 'a') as f:
                f.write(json.dumps(asdict(event)) + '\n')

        return event

    def apply_oom_fallback(self, error_message: str = "") -> Optional[RunConfig]:
        """Apply OOM fallback strategy.

        Fallback priority:
        1. Reduce eval_batch_size
        2. Reduce batch_size
        3. Reduce max_seq_length
        4. Reduce num_layers

        Returns:
            Updated config if fallback was possible, None if no more fallbacks available
        """
        if not self.config:
            raise RuntimeError("No active run")

        if not self.config.enable_oom_fallback:
            return None

        # Try fallbacks in order
        new_config = self.config.to_dict()
        reason = ""

        # 1. Reduce eval_batch_size first
        if self.config.eval_batch_size > 1:
            new_config["eval_batch_size"] = max(1, self.config.eval_batch_size // 2)
            reason = f"Reduced eval_batch_size from {self.config.eval_batch_size} to {new_config['eval_batch_size']}"

        # 2. Reduce batch_size
        elif self.config.batch_size > 1:
            new_config["batch_size"] = max(1, self.config.batch_size // 2)
            reason = f"Reduced batch_size from {self.config.batch_size} to {new_config['batch_size']}"

        # 3. Reduce max_seq_length
        elif self.config.max_seq_length > 512:
            new_seq_len = max(512, self.config.max_seq_length // 2)
            new_seq_len = (new_seq_len // 256) * 256  # Round to 256
            new_config["max_seq_length"] = new_seq_len
            reason = f"Reduced max_seq_length from {self.config.max_seq_length} to {new_seq_len}"

        # 4. Reduce num_layers
        elif self.config.num_layers > 4:
            new_config["num_layers"] = max(4, self.config.num_layers // 2)
            reason = f"Reduced num_layers from {self.config.num_layers} to {new_config['num_layers']}"

        # No more fallbacks available
        else:
            return None

        # Record the event
        self.record_oom_fallback(error_message, new_config, reason)

        # Apply new config
        self.config = RunConfig.from_dict(new_config)

        # Save updated config
        if self.current_run_dir:
            self.config.save(self.current_run_dir / "config" / "run_config.json")

        print(f"\n[OOM Fallback #{self.config.oom_fallback_count}] {reason}")

        return self.config

    def get_subdirectory(self, name: str) -> Path:
        """Get path to a subdirectory within the current run."""
        if not self.current_run_dir:
            raise RuntimeError("No active run")

        subdir = self.current_run_dir / name
        subdir.mkdir(exist_ok=True)
        return subdir

    def save_stats(self, name: str, data: Dict[str, Any]):
        """Save statistics to the stats directory."""
        stats_dir = self.get_subdirectory("stats")
        with open(stats_dir / f"{name}.json", 'w') as f:
            json.dump(data, f, indent=2)

    def get_config_summary(self) -> str:
        """Get a formatted summary of the current configuration."""
        if not self.config:
            return "No configuration loaded"

        lines = [
            "=" * 60,
            "Training Configuration Summary",
            "=" * 60,
            f"\nRun ID: {self.config.run_id}",
            f"Model: {self.config.model_id}",
            f"\nTraining Parameters:",
            f"  batch_size: {self.config.batch_size}",
            f"  max_seq_length: {self.config.max_seq_length}",
            f"  num_layers: {self.config.num_layers}",
            f"  learning_rate: {self.config.learning_rate}",
            f"  iters: {self.config.iters}",
            f"  grad_accum_steps: {self.config.grad_accum_steps}",
            f"\nLoRA Parameters:",
            f"  lora_rank: {self.config.lora_rank}",
            f"  lora_alpha: {self.config.lora_alpha}",
            f"  lora_dropout: {self.config.lora_dropout}",
            f"\nEvaluation:",
            f"  eval_batch_size: {self.config.eval_batch_size}",
            f"  eval_interval: {self.config.eval_interval}",
            f"\nData:",
            f"  training_data: {self.config.training_data}",
            f"  preprocess_enabled: {self.config.preprocess_enabled}",
            f"\nOOM Handling:",
            f"  enable_oom_fallback: {self.config.enable_oom_fallback}",
            f"  oom_fallback_count: {self.config.oom_fallback_count}",
            f"\nSeed: {self.config.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_config_summary(self):
        """Print configuration summary."""
        print(self.get_config_summary())

    def list_runs(self, limit: int = 10) -> List[Dict]:
        """List recent runs."""
        runs = []
        if not self.base_dir.exists():
            return runs

        for run_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        metadata["path"] = str(run_dir)
                        runs.append(metadata)

                if len(runs) >= limit:
                    break

        return runs

    def cleanup_incomplete_runs(self, keep_last: int = 5):
        """Remove incomplete or failed runs, keeping the last N."""
        runs = self.list_runs(limit=100)
        incomplete = [r for r in runs if r.get("status") not in ["completed", "training"]]

        for run in incomplete[keep_last:]:
            run_path = Path(run["path"])
            if run_path.exists():
                shutil.rmtree(run_path)
                print(f"Removed incomplete run: {run_path.name}")


if __name__ == "__main__":
    # Test the run manager
    manager = RunManager("./test_runs")

    config = RunConfig(
        model_id="mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        batch_size=2,
        max_seq_length=1024,
    )

    run_dir = manager.create_run(config, description="Test run")
    print(f"Created run: {run_dir}")

    manager.print_config_summary()

    # Test OOM fallback
    new_config = manager.apply_oom_fallback("Metal OOM error")
    if new_config:
        print("\nAfter fallback:")
        manager.print_config_summary()

    # Cleanup
    import shutil
    shutil.rmtree("./test_runs")
