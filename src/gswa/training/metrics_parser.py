"""
Training Metrics Parser - Parses MLX training output for real-time metrics.

Extracts metrics from mlx_lm training output:
- Training loss
- Tokens per second
- Iterations per second
- Memory usage (if available)
"""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class TrainingMetric:
    """A single training metric snapshot."""
    step: int
    timestamp: str
    train_loss: float
    tokens_per_sec: float = 0.0
    iters_per_sec: float = 0.0
    learning_rate: float = 0.0
    trained_tokens: int = 0
    peak_memory_gb: float = 0.0
    raw_line: str = ""


class MLXMetricsParser:
    """Parses MLX training output for metrics."""

    # MLX output patterns
    # Example: "Iter 10: Val loss 2.345, Val took 1.23s"
    # Example: "Iter 10: Train loss 2.123, Learning Rate 1.00e-05, Iter/sec 1.23, Tokens/sec 456.7"
    ITER_PATTERN = re.compile(
        r'Iter\s+(\d+):\s*'
        r'(?:Train\s+)?[Ll]oss\s+([\d.]+).*?'
        r'(?:Learning\s+Rate\s+([\d.e+-]+))?.*?'
        r'(?:Iter/sec\s+([\d.]+))?.*?'
        r'(?:[Tt]ok(?:en)?s?/sec[:\s]+([\d.]+))?',
        re.IGNORECASE
    )

    # Validation pattern
    VAL_PATTERN = re.compile(
        r'Iter\s+(\d+):\s*Val\s+loss\s+([\d.]+)',
        re.IGNORECASE
    )

    # Memory pattern (if MLX reports it)
    MEMORY_PATTERN = re.compile(
        r'(?:Peak\s+)?[Mm]emory[:\s]+([\d.]+)\s*(?:GB|MB)',
        re.IGNORECASE
    )

    def __init__(self, log_dir: str = None):
        """Initialize parser.

        Args:
            log_dir: Directory to save structured logs
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics: List[TrainingMetric] = []
        self.eval_metrics: List[Dict] = []
        self.start_time = datetime.now()

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._train_log = open(self.log_dir / "train_steps.jsonl", 'w')
            self._eval_log = open(self.log_dir / "eval_steps.jsonl", 'w')
        else:
            self._train_log = None
            self._eval_log = None

    def close(self):
        """Close log files."""
        if self._train_log:
            self._train_log.close()
        if self._eval_log:
            self._eval_log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def parse_line(self, line: str) -> Optional[TrainingMetric]:
        """Parse a single line of training output.

        Args:
            line: Training output line

        Returns:
            TrainingMetric if metrics found, None otherwise
        """
        line = line.strip()
        if not line:
            return None

        # Try to match training iteration
        match = self.ITER_PATTERN.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            lr = float(match.group(3)) if match.group(3) else 0.0
            iters_sec = float(match.group(4)) if match.group(4) else 0.0
            tokens_sec = float(match.group(5)) if match.group(5) else 0.0

            # Check for memory info
            mem_match = self.MEMORY_PATTERN.search(line)
            memory_gb = float(mem_match.group(1)) if mem_match else 0.0
            if "MB" in line and mem_match:
                memory_gb /= 1024

            metric = TrainingMetric(
                step=step,
                timestamp=datetime.now().isoformat(),
                train_loss=loss,
                tokens_per_sec=tokens_sec,
                iters_per_sec=iters_sec,
                learning_rate=lr,
                peak_memory_gb=memory_gb,
                raw_line=line,
            )

            self.metrics.append(metric)

            # Save to log file
            if self._train_log:
                self._train_log.write(json.dumps(asdict(metric)) + '\n')
                self._train_log.flush()

            return metric

        # Try to match validation
        val_match = self.VAL_PATTERN.search(line)
        if val_match:
            step = int(val_match.group(1))
            val_loss = float(val_match.group(2))

            eval_entry = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "eval_loss": val_loss,
                "eval_duration_sec": 0,
                "eval_samples": 0,
            }

            self.eval_metrics.append(eval_entry)

            if self._eval_log:
                self._eval_log.write(json.dumps(eval_entry) + '\n')
                self._eval_log.flush()

        return None

    def get_latest(self) -> Optional[TrainingMetric]:
        """Get the latest training metric."""
        return self.metrics[-1] if self.metrics else None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics:
            return {}

        losses = [m.train_loss for m in self.metrics]
        throughputs = [m.tokens_per_sec for m in self.metrics if m.tokens_per_sec > 0]
        eval_losses = [e["eval_loss"] for e in self.eval_metrics]

        return {
            "total_steps": len(self.metrics),
            "train_loss": {
                "final": losses[-1],
                "min": min(losses),
                "mean": sum(losses) / len(losses),
            },
            "throughput": {
                "mean": sum(throughputs) / len(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0,
            },
            "eval_loss": {
                "best": min(eval_losses) if eval_losses else None,
                "final": eval_losses[-1] if eval_losses else None,
            },
        }

    def print_progress(self, total_iters: int = None):
        """Print a formatted progress line."""
        metric = self.get_latest()
        if not metric:
            return

        progress = ""
        if total_iters:
            pct = metric.step / total_iters * 100
            progress = f"[{pct:5.1f}%] "

        status = f"{progress}Step {metric.step}: Loss={metric.train_loss:.4f}"

        if metric.tokens_per_sec > 0:
            status += f" | {metric.tokens_per_sec:.1f} tok/s"

        if metric.peak_memory_gb > 0:
            status += f" | {metric.peak_memory_gb:.1f}GB"

        print(f"\r{status}    ", end="", flush=True)

    def print_final_summary(self):
        """Print final training summary."""
        summary = self.get_summary()
        if not summary:
            return

        print("\n" + "-" * 50)
        print("Training Metrics Summary")
        print("-" * 50)
        print(f"Total Steps: {summary.get('total_steps', 0)}")

        train = summary.get("train_loss", {})
        print(f"\nTraining Loss:")
        print(f"  Final: {train.get('final', 0):.4f}")
        print(f"  Min:   {train.get('min', 0):.4f}")
        print(f"  Mean:  {train.get('mean', 0):.4f}")

        eval_loss = summary.get("eval_loss", {})
        if eval_loss.get("best"):
            print(f"\nEvaluation Loss:")
            print(f"  Best:  {eval_loss.get('best'):.4f}")
            print(f"  Final: {eval_loss.get('final'):.4f}")

        throughput = summary.get("throughput", {})
        if throughput.get("mean", 0) > 0:
            print(f"\nThroughput:")
            print(f"  Mean: {throughput.get('mean', 0):.1f} tok/s")
            print(f"  Max:  {throughput.get('max', 0):.1f} tok/s")


def create_ascii_loss_graph(metrics: List[TrainingMetric], width: int = 50, height: int = 10) -> str:
    """Create ASCII art loss graph.

    Args:
        metrics: List of training metrics
        width: Graph width in characters
        height: Graph height in rows

    Returns:
        ASCII graph string
    """
    if not metrics:
        return "No data to plot"

    losses = [m.train_loss for m in metrics]

    # Subsample if too many points
    if len(losses) > width:
        indices = [int(i * len(losses) / width) for i in range(width)]
        losses = [losses[i] for i in indices]

    min_loss = min(losses)
    max_loss = max(losses)
    range_loss = max_loss - min_loss if max_loss != min_loss else 1

    lines = []
    lines.append(f"Loss: {min_loss:.3f} - {max_loss:.3f}")
    lines.append("┌" + "─" * (len(losses) + 2) + "┐")

    for row in range(height - 1, -1, -1):
        threshold = min_loss + (row / (height - 1)) * range_loss
        line = "│ "
        for loss in losses:
            if loss >= threshold:
                line += "█"
            else:
                line += " "
        line += " │"
        lines.append(line)

    lines.append("└" + "─" * (len(losses) + 2) + "┘")
    lines.append(f" Step 0{' ' * (len(losses) - 10)}Step {len(metrics)}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample MLX output
    sample_output = """
    Iter 1: Train loss 2.123, Learning Rate 1.00e-05, It/sec 1.23, Tokens/sec 456.7
    Iter 2: Train loss 2.056, Learning Rate 1.00e-05, It/sec 1.25, Tokens/sec 460.2
    Iter 3: Train loss 1.987, Learning Rate 1.00e-05, It/sec 1.24, Tokens/sec 458.9
    Iter 4: Val loss 1.890, Val took 2.34s
    Iter 5: Train loss 1.923, Learning Rate 1.00e-05, It/sec 1.26, Tokens/sec 462.1
    """

    parser = MLXMetricsParser()

    for line in sample_output.strip().split('\n'):
        metric = parser.parse_line(line)
        if metric:
            parser.print_progress(total_iters=100)

    print("\n")
    parser.print_final_summary()

    print("\nASCII Loss Graph:")
    print(create_ascii_loss_graph(parser.metrics))
