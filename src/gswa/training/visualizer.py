"""
Training Visualizer Module - Generates plots and reports.

Provides:
- Training loss and eval loss plots
- Throughput and memory plots
- Length distribution histograms
- HTML report generation
- PNG export for all plots
"""

import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class TrainingVisualizer:
    """Generates visualizations for training runs."""

    # Color scheme
    COLORS = {
        "train_loss": "#2196F3",  # Blue
        "eval_loss": "#4CAF50",   # Green
        "throughput": "#FF9800",  # Orange
        "memory": "#9C27B0",      # Purple
        "oom": "#F44336",         # Red
        "background": "#FAFAFA",
        "grid": "#E0E0E0",
    }

    def __init__(self, log_dir: str = None):
        """Initialize visualizer.

        Args:
            log_dir: Directory containing training logs
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.data = {}
        self.preprocess_stats = {}

        if self.log_dir:
            self._load_data()

    def _load_data(self):
        """Load training data from log files."""
        if not self.log_dir or not self.log_dir.exists():
            return

        # Load training steps
        train_log = self.log_dir / "train_steps.jsonl"
        if train_log.exists():
            steps = []
            with open(train_log, 'r') as f:
                for line in f:
                    if line.strip():
                        steps.append(json.loads(line))

            if steps:
                self.data["steps"] = [s["step"] for s in steps]
                self.data["train_loss"] = [s["train_loss"] for s in steps]
                self.data["learning_rate"] = [s["learning_rate"] for s in steps]
                self.data["tokens_per_sec"] = [s["tokens_per_sec"] for s in steps]
                self.data["peak_memory_gb"] = [s.get("peak_memory_gb", 0) for s in steps]
                self.data["elapsed_sec"] = [s.get("elapsed_sec", 0) for s in steps]

        # Load eval steps
        eval_log = self.log_dir / "eval_steps.jsonl"
        if eval_log.exists():
            eval_steps = []
            with open(eval_log, 'r') as f:
                for line in f:
                    if line.strip():
                        eval_steps.append(json.loads(line))

            if eval_steps:
                self.data["eval_steps"] = [s["step"] for s in eval_steps]
                self.data["eval_loss"] = [s["eval_loss"] for s in eval_steps]

        # Load events for OOM markers
        events_log = self.log_dir / "events.jsonl"
        if events_log.exists():
            self.data["oom_steps"] = []
            with open(events_log, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get("type") == "oom_fallback":
                            self.data["oom_steps"].append(event["step"])

        # Load preprocessing stats
        stats_dir = self.log_dir.parent / "stats"
        preprocess_file = stats_dir / "preprocess_stats.json"
        if preprocess_file.exists():
            with open(preprocess_file, 'r') as f:
                self.preprocess_stats = json.load(f)

    def set_data(self, data: Dict[str, List]):
        """Set data directly instead of loading from files."""
        self.data = data

    def set_preprocess_stats(self, stats: Dict[str, Any]):
        """Set preprocessing statistics."""
        self.preprocess_stats = stats

    def plot_loss(self, output_path: str = None, show: bool = False) -> Optional[str]:
        """Plot training and evaluation loss.

        Args:
            output_path: Path to save PNG
            show: Whether to show the plot (for notebooks)

        Returns:
            Base64 encoded PNG if no output_path, else None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping plot")
            return None

        if "steps" not in self.data or "train_loss" not in self.data:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.COLORS["background"])
        ax.set_facecolor(self.COLORS["background"])

        # Plot training loss
        ax.plot(
            self.data["steps"],
            self.data["train_loss"],
            color=self.COLORS["train_loss"],
            label="Train Loss",
            linewidth=1.5,
            alpha=0.8,
        )

        # Add smoothed line
        if len(self.data["train_loss"]) > 20:
            window = min(50, len(self.data["train_loss"]) // 10)
            smoothed = self._moving_average(self.data["train_loss"], window)
            ax.plot(
                self.data["steps"][:len(smoothed)],
                smoothed,
                color=self.COLORS["train_loss"],
                label="Train Loss (smoothed)",
                linewidth=2,
            )

        # Plot eval loss
        if "eval_steps" in self.data and "eval_loss" in self.data:
            ax.plot(
                self.data["eval_steps"],
                self.data["eval_loss"],
                color=self.COLORS["eval_loss"],
                label="Eval Loss",
                marker='o',
                markersize=6,
                linewidth=2,
            )

        # Mark OOM events
        if "oom_steps" in self.data:
            for oom_step in self.data["oom_steps"]:
                ax.axvline(
                    x=oom_step,
                    color=self.COLORS["oom"],
                    linestyle='--',
                    alpha=0.7,
                    label='OOM Fallback' if oom_step == self.data["oom_steps"][0] else None,
                )

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Progress", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, color=self.COLORS["grid"])

        plt.tight_layout()

        return self._save_or_encode(fig, output_path, show)

    def plot_throughput(self, output_path: str = None, show: bool = False) -> Optional[str]:
        """Plot tokens/sec throughput over training."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if "steps" not in self.data or "tokens_per_sec" not in self.data:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.COLORS["background"])
        ax.set_facecolor(self.COLORS["background"])

        ax.plot(
            self.data["steps"],
            self.data["tokens_per_sec"],
            color=self.COLORS["throughput"],
            linewidth=1,
            alpha=0.6,
        )

        # Smoothed line
        if len(self.data["tokens_per_sec"]) > 20:
            window = min(50, len(self.data["tokens_per_sec"]) // 10)
            smoothed = self._moving_average(self.data["tokens_per_sec"], window)
            ax.plot(
                self.data["steps"][:len(smoothed)],
                smoothed,
                color=self.COLORS["throughput"],
                linewidth=2,
                label=f"Tokens/sec (smoothed, window={window})",
            )

        # Mark OOM events
        if "oom_steps" in self.data:
            for oom_step in self.data["oom_steps"]:
                ax.axvline(x=oom_step, color=self.COLORS["oom"], linestyle='--', alpha=0.7)

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Tokens/sec", fontsize=12)
        ax.set_title("Training Throughput", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, color=self.COLORS["grid"])

        plt.tight_layout()

        return self._save_or_encode(fig, output_path, show)

    def plot_memory(self, output_path: str = None, show: bool = False) -> Optional[str]:
        """Plot peak memory usage over training."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if "steps" not in self.data or "peak_memory_gb" not in self.data:
            return None

        # Skip if no memory data
        if all(m == 0 for m in self.data["peak_memory_gb"]):
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.COLORS["background"])
        ax.set_facecolor(self.COLORS["background"])

        ax.fill_between(
            self.data["steps"],
            self.data["peak_memory_gb"],
            alpha=0.3,
            color=self.COLORS["memory"],
        )
        ax.plot(
            self.data["steps"],
            self.data["peak_memory_gb"],
            color=self.COLORS["memory"],
            linewidth=1.5,
            label="Peak Memory (GB)",
        )

        # Mark OOM events
        if "oom_steps" in self.data:
            for oom_step in self.data["oom_steps"]:
                ax.axvline(x=oom_step, color=self.COLORS["oom"], linestyle='--', alpha=0.7)

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Memory (GB)", fontsize=12)
        ax.set_title("Memory Usage", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, color=self.COLORS["grid"])

        plt.tight_layout()

        return self._save_or_encode(fig, output_path, show)

    def plot_length_distribution(
        self,
        output_path: str = None,
        show: bool = False
    ) -> Optional[str]:
        """Plot token length distribution before/after preprocessing."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not self.preprocess_stats:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(self.COLORS["background"])

        # Before preprocessing
        ax1 = axes[0]
        ax1.set_facecolor(self.COLORS["background"])

        thresholds = [512, 1024, 2048, 4096]
        before_values = [
            self.preprocess_stats.get(f"over_{t}_before", 0)
            for t in thresholds
        ]
        total_before = self.preprocess_stats.get("total_entries_before", 1)

        ax1.bar(
            [f">{t}" for t in thresholds],
            [v / total_before * 100 for v in before_values],
            color=self.COLORS["oom"],
            alpha=0.7,
        )
        ax1.set_ylabel("Percentage of samples (%)", fontsize=11)
        ax1.set_title("Before Preprocessing", fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 100)

        # Add stats text
        stats_text = (
            f"Total: {total_before:,}\n"
            f"Max: {self.preprocess_stats.get('max_tokens_before', 0):,}\n"
            f"P99: {self.preprocess_stats.get('p99_tokens_before', 0):,}"
        )
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # After preprocessing
        ax2 = axes[1]
        ax2.set_facecolor(self.COLORS["background"])

        after_values = [
            self.preprocess_stats.get(f"over_{t}_after", 0)
            for t in thresholds
        ]
        total_after = self.preprocess_stats.get("total_entries_after", 1)

        ax2.bar(
            [f">{t}" for t in thresholds],
            [v / total_after * 100 for v in after_values],
            color=self.COLORS["eval_loss"],
            alpha=0.7,
        )
        ax2.set_ylabel("Percentage of samples (%)", fontsize=11)
        ax2.set_title("After Preprocessing", fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)

        # Add stats text
        stats_text = (
            f"Total: {total_after:,}\n"
            f"Max: {self.preprocess_stats.get('max_tokens_after', 0):,}\n"
            f"P99: {self.preprocess_stats.get('p99_tokens_after', 0):,}"
        )
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle("Token Length Distribution", fontsize=14, fontweight='bold')
        plt.tight_layout()

        return self._save_or_encode(fig, output_path, show)

    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average."""
        if len(data) < window:
            return data
        result = []
        for i in range(len(data) - window + 1):
            result.append(sum(data[i:i+window]) / window)
        return result

    def _save_or_encode(
        self,
        fig,
        output_path: str = None,
        show: bool = False
    ) -> Optional[str]:
        """Save figure to file or encode as base64."""
        if show:
            plt.show()
            plt.close(fig)
            return None

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None

        # Return base64 encoded
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')

    def generate_all_plots(self, output_dir: str):
        """Generate all plots and save to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Loss plot
        loss_path = output_dir / "loss.png"
        self.plot_loss(str(loss_path))
        if loss_path.exists():
            plots["loss"] = str(loss_path)

        # Throughput plot
        throughput_path = output_dir / "throughput.png"
        self.plot_throughput(str(throughput_path))
        if throughput_path.exists():
            plots["throughput"] = str(throughput_path)

        # Memory plot
        memory_path = output_dir / "memory.png"
        self.plot_memory(str(memory_path))
        if memory_path.exists():
            plots["memory"] = str(memory_path)

        # Length distribution
        length_path = output_dir / "length_distribution.png"
        self.plot_length_distribution(str(length_path))
        if length_path.exists():
            plots["length_distribution"] = str(length_path)

        return plots

    def generate_html_report(
        self,
        output_path: str,
        config: Dict[str, Any] = None,
        summary: Dict[str, Any] = None,
    ):
        """Generate an HTML report with embedded plots.

        Args:
            output_path: Path for HTML file
            config: Training configuration
            summary: Training summary metrics
        """
        # Generate base64 encoded plots
        loss_img = self.plot_loss()
        throughput_img = self.plot_throughput()
        memory_img = self.plot_memory()
        length_img = self.plot_length_distribution()

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Training Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: {self.COLORS['background']};
            color: #333;
        }}
        h1, h2, h3 {{
            color: #1a1a1a;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid {self.COLORS['grid']};
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid {self.COLORS['grid']};
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        .metric {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: {self.COLORS['train_loss']};
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        .timestamp {{
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Training Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

        # Summary metrics
        if summary:
            html += """
    <div class="section">
        <h2>Summary</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: center;">
"""
            metrics = [
                ("Steps", summary.get("total_steps", "N/A")),
                ("Tokens", f"{summary.get('total_tokens', 0):,}"),
                ("Duration", f"{summary.get('elapsed_sec', 0):.0f}s"),
                ("Final Loss", f"{summary.get('train_loss', {}).get('final', 'N/A'):.4f}" if summary.get('train_loss', {}).get('final') else "N/A"),
                ("Best Eval", f"{summary.get('eval_loss', {}).get('best', 'N/A'):.4f}" if summary.get('eval_loss', {}).get('best') else "N/A"),
                ("OOM Events", summary.get("oom_events", 0)),
            ]
            for label, value in metrics:
                html += f"""
            <div class="metric">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
"""
            html += """
        </div>
    </div>
"""

        # Configuration
        if config:
            html += """
    <div class="section">
        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
"""
            for key, value in config.items():
                if not key.startswith("_") and not isinstance(value, dict):
                    html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
            html += """
        </table>
    </div>
"""

        # Plots
        html += """
    <div class="section">
        <h2>Training Progress</h2>
"""
        if loss_img:
            html += f"""
        <div class="plot-container">
            <h3>Loss</h3>
            <img src="data:image/png;base64,{loss_img}" alt="Loss Plot">
        </div>
"""
        if throughput_img:
            html += f"""
        <div class="plot-container">
            <h3>Throughput</h3>
            <img src="data:image/png;base64,{throughput_img}" alt="Throughput Plot">
        </div>
"""
        if memory_img:
            html += f"""
        <div class="plot-container">
            <h3>Memory Usage</h3>
            <img src="data:image/png;base64,{memory_img}" alt="Memory Plot">
        </div>
"""
        html += """
    </div>
"""

        # Preprocessing
        if length_img:
            html += f"""
    <div class="section">
        <h2>Data Preprocessing</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{length_img}" alt="Length Distribution">
        </div>
    </div>
"""

        html += """
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)


def generate_training_report(
    log_dir: str,
    output_dir: str = None,
    config: Dict[str, Any] = None,
):
    """Convenience function to generate all visualizations.

    Args:
        log_dir: Directory containing training logs
        output_dir: Output directory for plots (default: log_dir/../reports)
        config: Training configuration
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir) if output_dir else log_dir.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = TrainingVisualizer(str(log_dir))

    # Generate plots
    plots = visualizer.generate_all_plots(str(output_dir / "plots"))
    print(f"Generated {len(plots)} plots")

    # Load summary if available
    summary_file = log_dir / "summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

    # Generate HTML report
    html_path = output_dir / "report.html"
    visualizer.generate_html_report(str(html_path), config, summary)
    print(f"Generated HTML report: {html_path}")

    return str(html_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training visualizations")
    parser.add_argument("--log-dir", "-l", required=True, help="Directory with training logs")
    parser.add_argument("--output-dir", "-o", help="Output directory for plots and report")
    parser.add_argument("--config", "-c", help="Config JSON file to include in report")

    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    report_path = generate_training_report(
        args.log_dir,
        args.output_dir,
        config,
    )
    print(f"\nReport generated: {report_path}")
