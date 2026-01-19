"""
Training Logger Module - Structured logging for training metrics.

Provides:
- JSONL structured logs for each step
- Real-time metric tracking
- OOM event logging
- Exportable metrics for visualization
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading


@dataclass
class TrainingStep:
    """A single training step record."""
    step: int
    timestamp: str
    train_loss: float
    learning_rate: float
    tokens_per_sec: float
    iters_per_sec: float
    trained_tokens: int
    peak_memory_gb: float
    batch_size: int
    seq_length: int
    grad_norm: Optional[float] = None
    elapsed_sec: float = 0.0


@dataclass
class EvalStep:
    """An evaluation step record."""
    step: int
    timestamp: str
    eval_loss: float
    eval_duration_sec: float
    eval_samples: int
    perplexity: Optional[float] = None


@dataclass
class OOMEvent:
    """OOM event record."""
    step: int
    timestamp: str
    error_message: str
    config_before: Dict[str, Any]
    config_after: Dict[str, Any]
    fallback_action: str


@dataclass
class ConfigChange:
    """Configuration change record."""
    step: int
    timestamp: str
    reason: str
    changes: Dict[str, Any]


class TrainingLogger:
    """Structured logger for training runs."""

    def __init__(
        self,
        log_dir: str,
        run_id: str = "",
        flush_interval: int = 10,
    ):
        """Initialize logger.

        Args:
            log_dir: Directory for log files
            run_id: Run identifier
            flush_interval: Flush to disk every N steps
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S')
        self.flush_interval = flush_interval

        # Log file paths
        self.train_log_path = self.log_dir / "train_steps.jsonl"
        self.eval_log_path = self.log_dir / "eval_steps.jsonl"
        self.events_log_path = self.log_dir / "events.jsonl"

        # In-memory buffers
        self.train_steps: List[TrainingStep] = []
        self.eval_steps: List[EvalStep] = []
        self.events: List[Dict] = []

        # Tracking
        self.start_time = time.time()
        self.current_step = 0
        self.total_tokens = 0
        self.best_eval_loss = float('inf')

        # Thread safety
        self._lock = threading.Lock()

        # Open log files
        self._train_file = open(self.train_log_path, 'a')
        self._eval_file = open(self.eval_log_path, 'a')
        self._events_file = open(self.events_log_path, 'a')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close log files."""
        self._train_file.close()
        self._eval_file.close()
        self._events_file.close()

    def log_step(
        self,
        step: int,
        train_loss: float,
        learning_rate: float,
        tokens_per_sec: float,
        trained_tokens: int,
        peak_memory_gb: float = 0.0,
        batch_size: int = 1,
        seq_length: int = 1024,
        grad_norm: float = None,
    ):
        """Log a training step.

        Args:
            step: Current training step
            train_loss: Training loss
            learning_rate: Current learning rate
            tokens_per_sec: Training throughput
            trained_tokens: Total tokens trained so far
            peak_memory_gb: Peak memory usage
            batch_size: Current batch size
            seq_length: Current sequence length
            grad_norm: Gradient norm (optional)
        """
        elapsed = time.time() - self.start_time

        record = TrainingStep(
            step=step,
            timestamp=datetime.now().isoformat(),
            train_loss=train_loss,
            learning_rate=learning_rate,
            tokens_per_sec=tokens_per_sec,
            iters_per_sec=step / elapsed if elapsed > 0 else 0,
            trained_tokens=trained_tokens,
            peak_memory_gb=peak_memory_gb,
            batch_size=batch_size,
            seq_length=seq_length,
            grad_norm=grad_norm,
            elapsed_sec=elapsed,
        )

        with self._lock:
            self.train_steps.append(record)
            self.current_step = step
            self.total_tokens = trained_tokens

            # Write to file
            self._train_file.write(json.dumps(asdict(record)) + '\n')

            # Periodic flush
            if step % self.flush_interval == 0:
                self._train_file.flush()

    def log_eval(
        self,
        step: int,
        eval_loss: float,
        eval_duration_sec: float,
        eval_samples: int,
        perplexity: float = None,
    ):
        """Log an evaluation step.

        Args:
            step: Current training step
            eval_loss: Evaluation loss
            eval_duration_sec: Time taken for evaluation
            eval_samples: Number of samples evaluated
            perplexity: Perplexity (optional)
        """
        record = EvalStep(
            step=step,
            timestamp=datetime.now().isoformat(),
            eval_loss=eval_loss,
            eval_duration_sec=eval_duration_sec,
            eval_samples=eval_samples,
            perplexity=perplexity,
        )

        with self._lock:
            self.eval_steps.append(record)

            # Track best
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss

            # Write to file
            self._eval_file.write(json.dumps(asdict(record)) + '\n')
            self._eval_file.flush()

    def log_oom_event(
        self,
        step: int,
        error_message: str,
        config_before: Dict[str, Any],
        config_after: Dict[str, Any],
        fallback_action: str,
    ):
        """Log an OOM fallback event.

        Args:
            step: Step where OOM occurred
            error_message: Error message
            config_before: Configuration before fallback
            config_after: Configuration after fallback
            fallback_action: Description of fallback taken
        """
        event = {
            "type": "oom_fallback",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
            "config_before": config_before,
            "config_after": config_after,
            "fallback_action": fallback_action,
        }

        with self._lock:
            self.events.append(event)
            self._events_file.write(json.dumps(event) + '\n')
            self._events_file.flush()

        print(f"\n[OOM Event @ step {step}] {fallback_action}")

    def log_config_change(
        self,
        step: int,
        reason: str,
        changes: Dict[str, Any],
    ):
        """Log a configuration change.

        Args:
            step: Step where change occurred
            reason: Reason for change
            changes: Dictionary of changes
        """
        event = {
            "type": "config_change",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "changes": changes,
        }

        with self._lock:
            self.events.append(event)
            self._events_file.write(json.dumps(event) + '\n')
            self._events_file.flush()

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a generic event.

        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            "type": event_type,
            "step": self.current_step,
            "timestamp": datetime.now().isoformat(),
            **data,
        }

        with self._lock:
            self.events.append(event)
            self._events_file.write(json.dumps(event) + '\n')
            self._events_file.flush()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        if not self.train_steps:
            return {}

        losses = [s.train_loss for s in self.train_steps]
        throughputs = [s.tokens_per_sec for s in self.train_steps]
        memories = [s.peak_memory_gb for s in self.train_steps if s.peak_memory_gb > 0]

        eval_losses = [s.eval_loss for s in self.eval_steps]

        return {
            "run_id": self.run_id,
            "total_steps": len(self.train_steps),
            "total_tokens": self.total_tokens,
            "elapsed_sec": time.time() - self.start_time,
            "train_loss": {
                "final": losses[-1] if losses else None,
                "min": min(losses) if losses else None,
                "mean": sum(losses) / len(losses) if losses else None,
            },
            "throughput": {
                "mean": sum(throughputs) / len(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0,
            },
            "memory": {
                "max": max(memories) if memories else 0,
                "mean": sum(memories) / len(memories) if memories else 0,
            },
            "eval_loss": {
                "best": self.best_eval_loss if self.eval_steps else None,
                "final": eval_losses[-1] if eval_losses else None,
            },
            "oom_events": sum(1 for e in self.events if e.get("type") == "oom_fallback"),
        }

    def export_for_plotting(self) -> Dict[str, List]:
        """Export data in format suitable for plotting.

        Returns:
            Dictionary with lists for each metric
        """
        return {
            "steps": [s.step for s in self.train_steps],
            "train_loss": [s.train_loss for s in self.train_steps],
            "learning_rate": [s.learning_rate for s in self.train_steps],
            "tokens_per_sec": [s.tokens_per_sec for s in self.train_steps],
            "peak_memory_gb": [s.peak_memory_gb for s in self.train_steps],
            "elapsed_sec": [s.elapsed_sec for s in self.train_steps],
            "eval_steps": [s.step for s in self.eval_steps],
            "eval_loss": [s.eval_loss for s in self.eval_steps],
            "oom_steps": [e["step"] for e in self.events if e.get("type") == "oom_fallback"],
        }

    def save_summary(self, path: str = None):
        """Save metrics summary to JSON."""
        path = path or (self.log_dir / "summary.json")
        summary = self.get_metrics_summary()
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    @classmethod
    def load_from_logs(cls, log_dir: str) -> "TrainingLogger":
        """Load a logger from existing log files.

        Args:
            log_dir: Directory containing log files

        Returns:
            TrainingLogger with loaded data
        """
        log_dir = Path(log_dir)
        logger = cls(str(log_dir), flush_interval=9999)

        # Load training steps
        train_log = log_dir / "train_steps.jsonl"
        if train_log.exists():
            with open(train_log, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        logger.train_steps.append(TrainingStep(**data))

        # Load eval steps
        eval_log = log_dir / "eval_steps.jsonl"
        if eval_log.exists():
            with open(eval_log, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        logger.eval_steps.append(EvalStep(**data))

        # Load events
        events_log = log_dir / "events.jsonl"
        if events_log.exists():
            with open(events_log, 'r') as f:
                for line in f:
                    if line.strip():
                        logger.events.append(json.loads(line))

        return logger


class LiveProgressBar:
    """Live progress bar for terminal output."""

    def __init__(self, total: int, desc: str = "Training"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(
        self,
        step: int,
        loss: float = None,
        tokens_per_sec: float = None,
        memory_gb: float = None,
    ):
        """Update progress bar."""
        self.current = step
        elapsed = time.time() - self.start_time

        # Calculate progress
        pct = step / self.total * 100 if self.total > 0 else 0
        eta = (elapsed / step * (self.total - step)) if step > 0 else 0

        # Build progress bar
        bar_width = 30
        filled = int(bar_width * step / self.total) if self.total > 0 else 0
        bar = '=' * filled + '>' + '.' * (bar_width - filled - 1)

        # Build status line
        status = f"\r{self.desc}: [{bar}] {pct:5.1f}% | Step {step}/{self.total}"

        if loss is not None:
            status += f" | Loss: {loss:.4f}"
        if tokens_per_sec is not None:
            status += f" | {tokens_per_sec:.1f} tok/s"
        if memory_gb is not None:
            status += f" | {memory_gb:.1f}GB"
        if eta > 0:
            status += f" | ETA: {eta:.0f}s"

        print(status, end='', flush=True)

    def finish(self, message: str = "Complete"):
        """Finish progress bar."""
        elapsed = time.time() - self.start_time
        print(f"\n{self.desc}: {message} in {elapsed:.1f}s")


if __name__ == "__main__":
    # Test the logger
    import tempfile
    import shutil

    test_dir = tempfile.mkdtemp()

    try:
        with TrainingLogger(test_dir, "test_run") as logger:
            # Simulate training
            for step in range(1, 101):
                logger.log_step(
                    step=step,
                    train_loss=2.5 - step * 0.02,
                    learning_rate=1e-5,
                    tokens_per_sec=100 + step,
                    trained_tokens=step * 1000,
                    peak_memory_gb=8.0 + step * 0.01,
                )

                if step % 20 == 0:
                    logger.log_eval(
                        step=step,
                        eval_loss=2.3 - step * 0.015,
                        eval_duration_sec=5.0,
                        eval_samples=100,
                    )

            # Simulate OOM
            logger.log_oom_event(
                step=50,
                error_message="Metal OOM",
                config_before={"batch_size": 2},
                config_after={"batch_size": 1},
                fallback_action="Reduced batch_size from 2 to 1",
            )

            logger.save_summary()

        # Test loading
        loaded = TrainingLogger.load_from_logs(test_dir)
        print(f"\nLoaded {len(loaded.train_steps)} training steps")
        print(f"Loaded {len(loaded.eval_steps)} eval steps")
        print(f"Loaded {len(loaded.events)} events")

        print("\nMetrics summary:")
        print(json.dumps(loaded.get_metrics_summary(), indent=2))

    finally:
        shutil.rmtree(test_dir)
