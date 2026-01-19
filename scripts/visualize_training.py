#!/usr/bin/env python3
"""
Training Visualization Script - Generate plots and reports from training runs.

Usage:
    # Visualize a specific run
    python scripts/visualize_training.py --run-dir runs/20240115-123456

    # Visualize the latest run
    python scripts/visualize_training.py --latest

    # Watch training in real-time
    python scripts/visualize_training.py --run-dir runs/20240115-123456 --watch

    # Generate only specific plots
    python scripts/visualize_training.py --run-dir runs/xxx --plots loss,throughput
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from gswa.training.visualizer import TrainingVisualizer, generate_training_report
    from gswa.training.logger import TrainingLogger
except ImportError:
    print("Error: gswa.training module not found. Make sure you're in the project root.")
    sys.exit(1)


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    if not runs_dir.exists():
        return None

    runs = sorted(runs_dir.iterdir(), reverse=True)
    for run in runs:
        if run.is_dir() and (run / "logs").exists():
            return run
    return None


def print_live_metrics(log_dir: Path):
    """Print live training metrics."""
    train_log = log_dir / "train_steps.jsonl"
    eval_log = log_dir / "eval_steps.jsonl"

    if not train_log.exists():
        print("Waiting for training to start...")
        return None

    # Read latest training step
    latest_train = None
    with open(train_log, 'r') as f:
        for line in f:
            if line.strip():
                latest_train = json.loads(line)

    # Read latest eval step
    latest_eval = None
    if eval_log.exists():
        with open(eval_log, 'r') as f:
            for line in f:
                if line.strip():
                    latest_eval = json.loads(line)

    if latest_train:
        step = latest_train.get("step", 0)
        train_loss = latest_train.get("train_loss", 0)
        tokens_per_sec = latest_train.get("tokens_per_sec", 0)
        peak_mem = latest_train.get("peak_memory_gb", 0)
        elapsed = latest_train.get("elapsed_sec", 0)

        print(f"\r[Step {step:5d}] "
              f"Loss: {train_loss:.4f} | "
              f"Tok/s: {tokens_per_sec:6.1f} | "
              f"Mem: {peak_mem:5.1f}GB | "
              f"Time: {elapsed/60:5.1f}min", end="")

        if latest_eval:
            eval_loss = latest_eval.get("eval_loss", 0)
            print(f" | Eval: {eval_loss:.4f}", end="")

        print("    ", end="", flush=True)

    return latest_train


def watch_training(log_dir: Path, interval: float = 2.0):
    """Watch training progress in real-time."""
    print("\n" + "=" * 60)
    print("Live Training Monitor")
    print("=" * 60)
    print("Press Ctrl+C to stop watching\n")

    last_step = -1

    try:
        while True:
            latest = print_live_metrics(log_dir)
            if latest and latest.get("step", 0) != last_step:
                last_step = latest.get("step", 0)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nStopped watching.")


def print_training_summary(log_dir: Path):
    """Print a summary of the training run."""
    logger = TrainingLogger.load_from_logs(str(log_dir))
    summary = logger.get_metrics_summary()

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    print(f"\nRun ID: {summary.get('run_id', 'N/A')}")
    print(f"Total Steps: {summary.get('total_steps', 0):,}")
    print(f"Total Tokens: {summary.get('total_tokens', 0):,}")
    print(f"Duration: {summary.get('elapsed_sec', 0)/60:.1f} minutes")

    train_loss = summary.get("train_loss", {})
    print(f"\nTraining Loss:")
    print(f"  Final: {train_loss.get('final', 'N/A'):.4f}" if train_loss.get('final') else "  Final: N/A")
    print(f"  Min: {train_loss.get('min', 'N/A'):.4f}" if train_loss.get('min') else "  Min: N/A")
    print(f"  Mean: {train_loss.get('mean', 'N/A'):.4f}" if train_loss.get('mean') else "  Mean: N/A")

    eval_loss = summary.get("eval_loss", {})
    if eval_loss.get("best"):
        print(f"\nEvaluation Loss:")
        print(f"  Best: {eval_loss.get('best'):.4f}")
        print(f"  Final: {eval_loss.get('final', 'N/A'):.4f}" if eval_loss.get('final') else "  Final: N/A")

    throughput = summary.get("throughput", {})
    print(f"\nThroughput:")
    print(f"  Mean: {throughput.get('mean', 0):.1f} tok/s")
    print(f"  Max: {throughput.get('max', 0):.1f} tok/s")

    memory = summary.get("memory", {})
    if memory.get("max", 0) > 0:
        print(f"\nMemory:")
        print(f"  Max: {memory.get('max', 0):.1f} GB")
        print(f"  Mean: {memory.get('mean', 0):.1f} GB")

    oom_events = summary.get("oom_events", 0)
    if oom_events > 0:
        print(f"\nOOM Events: {oom_events}")


def generate_ascii_loss_plot(log_dir: Path, width: int = 60, height: int = 15):
    """Generate ASCII art loss plot for terminal."""
    train_log = log_dir / "train_steps.jsonl"
    if not train_log.exists():
        return

    # Load data
    steps = []
    losses = []
    with open(train_log, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                steps.append(data["step"])
                losses.append(data["train_loss"])

    if not losses:
        return

    # Subsample for display
    if len(losses) > width:
        indices = [int(i * len(losses) / width) for i in range(width)]
        losses = [losses[i] for i in indices]
        steps = [steps[i] for i in indices]

    # Normalize
    min_loss = min(losses)
    max_loss = max(losses)
    range_loss = max_loss - min_loss if max_loss != min_loss else 1

    # Create plot
    print("\n" + "=" * 60)
    print("Training Loss (ASCII)")
    print("=" * 60)

    # Y-axis labels
    for row in range(height, -1, -1):
        y_val = min_loss + (row / height) * range_loss
        label = f"{y_val:6.3f} |"

        line = ""
        for col, loss in enumerate(losses):
            normalized = (loss - min_loss) / range_loss * height
            if abs(normalized - row) < 0.5:
                line += "*"
            elif row == 0:
                line += "-"
            else:
                line += " "

        print(label + line)

    # X-axis
    print(" " * 8 + "+" + "-" * len(losses))
    print(f" " * 8 + f"0{' ' * (len(losses) - 10)}Step {steps[-1] if steps else 0}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize latest run
    python scripts/visualize_training.py --latest

    # Watch training in real-time
    python scripts/visualize_training.py --run-dir runs/xxx --watch

    # Generate HTML report only
    python scripts/visualize_training.py --run-dir runs/xxx --html

    # Show ASCII plot in terminal
    python scripts/visualize_training.py --run-dir runs/xxx --ascii
        """
    )

    parser.add_argument("--run-dir", "-r", help="Run directory to visualize")
    parser.add_argument("--latest", action="store_true", help="Use the latest run")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch training in real-time")
    parser.add_argument("--interval", type=float, default=2.0, help="Watch interval in seconds")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--ascii", action="store_true", help="Show ASCII loss plot")
    parser.add_argument("--summary", action="store_true", help="Show training summary")
    parser.add_argument("--plots", help="Comma-separated list of plots: loss,throughput,memory,length")
    parser.add_argument("--output-dir", "-o", help="Output directory for plots")

    args = parser.parse_args()

    # Find run directory
    runs_dir = Path("runs")

    if args.latest:
        run_dir = find_latest_run(runs_dir)
        if not run_dir:
            print("No runs found in ./runs/")
            return 1
        print(f"Using latest run: {run_dir.name}")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Default to latest
        run_dir = find_latest_run(runs_dir)
        if not run_dir:
            print("No runs found. Specify --run-dir or use --latest")
            return 1
        print(f"Using latest run: {run_dir.name}")

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 1

    log_dir = run_dir / "logs"
    if not log_dir.exists():
        print(f"Logs directory not found: {log_dir}")
        return 1

    # Watch mode
    if args.watch:
        watch_training(log_dir, args.interval)
        return 0

    # ASCII plot
    if args.ascii:
        generate_ascii_loss_plot(log_dir)
        print_training_summary(log_dir)
        return 0

    # Summary only
    if args.summary:
        print_training_summary(log_dir)
        return 0

    # Generate visualizations
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reports"

    print(f"\nGenerating visualizations...")
    print(f"  Log dir: {log_dir}")
    print(f"  Output dir: {output_dir}")

    # Load config if available
    config = None
    config_file = run_dir / "config" / "run_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    try:
        if args.html or not args.plots:
            # Generate full HTML report
            report_path = generate_training_report(str(log_dir), str(output_dir), config)
            print(f"\nHTML report generated: {report_path}")

        if args.plots:
            # Generate specific plots
            visualizer = TrainingVisualizer(str(log_dir))
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            plot_types = args.plots.split(",")
            for plot_type in plot_types:
                plot_type = plot_type.strip().lower()
                if plot_type == "loss":
                    visualizer.plot_loss(str(plots_dir / "loss.png"))
                    print(f"  Generated: {plots_dir / 'loss.png'}")
                elif plot_type == "throughput":
                    visualizer.plot_throughput(str(plots_dir / "throughput.png"))
                    print(f"  Generated: {plots_dir / 'throughput.png'}")
                elif plot_type == "memory":
                    visualizer.plot_memory(str(plots_dir / "memory.png"))
                    print(f"  Generated: {plots_dir / 'memory.png'}")
                elif plot_type == "length":
                    visualizer.plot_length_distribution(str(plots_dir / "length_distribution.png"))
                    print(f"  Generated: {plots_dir / 'length_distribution.png'}")

        # Also print summary
        print_training_summary(log_dir)

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        # Still show ASCII plot as fallback
        generate_ascii_loss_plot(log_dir)
        print_training_summary(log_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
