"""Model Registry Service.

Auto-discovers trained LoRA adapters in the models/ directory
and provides model selection for inference.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from gswa.config import get_settings

logger = logging.getLogger(__name__)


# Known base model mappings: short_name -> HuggingFace model ID
BASE_MODEL_MAP = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407",
    "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "gemma": "google/gemma-2-9b-it",
}

# Human-readable names for base models
BASE_MODEL_FRIENDLY_NAMES = {
    "mistral": "Mistral 7B",
    "mistral-nemo": "Mistral Nemo 12B",
    "Mistral": "Mistral Nemo 12B",  # Handle capitalized version from training
    "mistral-large": "Mistral Large 123B",
    "llama3.3": "Llama 3.3 70B",
    "llama3-8b": "Llama 3.1 8B",
    "qwen": "Qwen 2.5 7B",
    "qwen-14b": "Qwen 2.5 14B",
    "phi": "Phi 3.5 Mini",
    "gemma": "Gemma 2 9B",
}


class TrainedModel:
    """Represents a trained LoRA adapter with its metadata."""

    def __init__(self, model_dir: Path, config: dict):
        self.model_dir = model_dir
        self.config = config
        self.name = model_dir.name  # e.g. "gswa-lora-Mistral-20260123-012408"
        self.base_model = config.get("base_model", "")
        self.model_short = config.get("model_short", "unknown")
        self.quantization = config.get("quantization", "4bit")
        self.lora_r = config.get("lora_r", 16)
        self.epochs = config.get("epochs", 3)
        self.started_at = config.get("started_at", "")
        self.completed_at = config.get("completed_at", "")

    @property
    def display_name(self) -> str:
        """Human-readable display name with key parameters for distinction."""
        # Get friendly model name
        friendly_base = BASE_MODEL_FRIENDLY_NAMES.get(
            self.model_short,
            BASE_MODEL_FRIENDLY_NAMES.get(self.model_short.lower(), self.model_short)
        )

        # Extract date from directory name (format: gswa-lora-{ModelShort}-{YYYYMMDD}-{HHMMSS})
        parts = self.name.split("-")
        date_str = ""
        for part in parts:
            if len(part) == 8 and part.isdigit():
                # Format as "Jan 26" instead of "2026-01-26"
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                try:
                    month_idx = int(part[4:6]) - 1
                    day = int(part[6:8])
                    date_str = f"{month_names[month_idx]} {day}"
                except (ValueError, IndexError):
                    date_str = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
                break

        # Check for suffix like "-3ep" in directory name
        suffix = ""
        if self.name.endswith("-3ep"):
            suffix = " (3ep checkpoint)"
        elif self.name.endswith("-2ep"):
            suffix = " (2ep checkpoint)"

        # Include key distinguishing parameters: r=rank, epochs
        params = f"r{self.lora_r} {self.epochs}ep"

        return f"{friendly_base} ({date_str}) {params}{suffix}"

    @property
    def adapter_path(self) -> str:
        """Path to the LoRA adapter weights."""
        return str(self.model_dir)

    @property
    def vllm_lora_name(self) -> str:
        """Name used for vLLM --lora-modules parameter."""
        return self.name

    @property
    def description(self) -> str:
        """Human-readable description with training details."""
        friendly_base = BASE_MODEL_FRIENDLY_NAMES.get(
            self.model_short,
            BASE_MODEL_FRIENDLY_NAMES.get(self.model_short.lower(), self.model_short)
        )
        # Provide detailed description
        desc = f"Fine-tuned {friendly_base}"
        desc += f" | LoRA r={self.lora_r}"
        desc += f" | {self.epochs} epochs"
        desc += f" | {self.quantization}"
        if self.started_at:
            desc += f" | Trained: {self.started_at[:10]}"
        return desc

    @property
    def short_params(self) -> str:
        """Short parameter summary for dropdown display."""
        return f"r{self.lora_r} {self.epochs}ep"

    def to_dict(self, is_recommended: bool = False) -> dict:
        """Serialize to dictionary for API response."""
        return {
            "id": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "short_params": self.short_params,
            "base_model": self.base_model,
            "model_short": self.model_short,
            "adapter_path": self.adapter_path,
            "quantization": self.quantization,
            "lora_r": self.lora_r,
            "epochs": self.epochs,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "is_recommended": is_recommended,
        }


class ModelRegistry:
    """Registry that discovers and manages trained LoRA models."""

    def __init__(self, models_dir: Optional[str] = None):
        self.settings = get_settings()
        self.models_dir = Path(models_dir or self.settings.models_dir)
        self._models: dict[str, TrainedModel] = {}
        self._scan()

    def _scan(self) -> None:
        """Scan models directory for trained adapters."""
        self._models.clear()

        if not self.models_dir.exists():
            logger.info(f"Models directory not found: {self.models_dir}")
            return

        for model_dir in sorted(self.models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if not model_dir.name.startswith("gswa-"):
                continue

            config_file = model_dir / "training_config.json"
            if not config_file.exists():
                continue

            # Check that adapter weights exist
            adapter_config = model_dir / "adapter_config.json"
            adapter_model = model_dir / "adapter_model.safetensors"
            adapter_model_bin = model_dir / "adapter_model.bin"
            if not adapter_config.exists():
                continue
            if not (adapter_model.exists() or adapter_model_bin.exists()):
                continue

            try:
                with open(config_file) as f:
                    config = json.load(f)
                model = TrainedModel(model_dir, config)
                self._models[model.name] = model
                logger.info(f"Discovered model: {model.display_name} ({model.name})")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping {model_dir.name}: {e}")

    def refresh(self) -> None:
        """Re-scan the models directory."""
        self._scan()

    @property
    def models(self) -> list[TrainedModel]:
        """Get all discovered models, newest first."""
        return sorted(
            self._models.values(),
            key=lambda m: m.started_at,
            reverse=True,
        )

    def get_model(self, model_id: str) -> Optional[TrainedModel]:
        """Get a specific model by ID."""
        return self._models.get(model_id)

    def get_latest(self, base_model_short: Optional[str] = None) -> Optional[TrainedModel]:
        """Get the latest trained model, optionally filtered by base model."""
        models = self.models
        if base_model_short:
            models = [m for m in models if m.model_short.lower() == base_model_short.lower()]
        return models[0] if models else None

    def get_models_by_base(self) -> dict[str, list[TrainedModel]]:
        """Group models by their base model."""
        groups: dict[str, list[TrainedModel]] = {}
        for model in self.models:
            key = model.model_short
            if key not in groups:
                groups[key] = []
            groups[key].append(model)
        return groups

    def get_vllm_lora_args(self) -> list[str]:
        """Generate vLLM --lora-modules arguments for all adapters of the same base model."""
        if not self._models:
            return []

        # Group by base model - vLLM can only serve one base model at a time
        # Return args for the most recent base model group
        groups = self.get_models_by_base()
        if not groups:
            return []

        # Use the base model with the most recent adapter
        latest = self.get_latest()
        if not latest:
            return []

        base_short = latest.model_short
        same_base = groups.get(base_short, [])

        args = []
        for model in same_base:
            args.append(f"{model.vllm_lora_name}={model.adapter_path}")
        return args

    def to_list(self) -> list[dict]:
        """Serialize all models to list of dicts for API response.

        The latest model (first in list) is marked as recommended.
        """
        models = self.models
        result = []
        for i, m in enumerate(models):
            # Mark the latest (newest) model as recommended
            is_recommended = (i == 0)
            result.append(m.to_dict(is_recommended=is_recommended))
        return result


# Singleton
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create model registry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_model_registry() -> None:
    """Reset the registry (for testing or after new training)."""
    global _registry
    _registry = None
