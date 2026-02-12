#!/usr/bin/env python3
"""
Plot training metrics from HuggingFace Trainer checkpoints.

Generates loss curves, learning rate schedules, and gradient norms.
Supports comparison mode for multiple training runs.

Usage:
    python scripts/plot_training.py models/gswa-lora-Mistral-20260130-*/
    python scripts/plot_training.py --compare models/gswa-lora-Mistral-*/
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    exit(1)


def find_trainer_state(model_dir: Path) -> dict | None:
    """Find and load trainer_state.json from model directory.

    Looks for training_metrics.json first (saved after training),
    then falls back to the latest checkpoint's trainer_state.json.

    Returns a dict with 'log_history' key.
    """
    model_dir = Path(model_dir)

    # Try training_metrics.json first (saved by finetune_lora.py)
    metrics_file = model_dir / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
            # Handle both formats: full state or just log_history
            if isinstance(data, list):
                return {"log_history": data}
            return data

    # Fall back to checkpoint trainer_state.json
    checkpoints = sorted(model_dir.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        return None

    # Use the last checkpoint (most complete training history)
    with open(checkpoints[-1]) as f:
        return json.load(f)


def load_training_config(model_dir: Path) -> dict:
    """Load training configuration from model directory."""
    config_file = model_dir / "training_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def extract_metrics(log_history: list) -> dict:
    """Extract training and eval metrics from log history."""
    train_losses = []
    eval_losses = []
    grad_norms = []
    learning_rates = []
    steps = []
    epochs = []
    eval_steps = []

    for entry in log_history:
        step = entry.get("step", 0)

        if "loss" in entry:  # Training log
            train_losses.append(entry["loss"])
            steps.append(step)
            epochs.append(entry.get("epoch", 0))

            if "grad_norm" in entry:
                grad_norms.append(entry["grad_norm"])
            if "learning_rate" in entry:
                learning_rates.append(entry["learning_rate"])

        elif "eval_loss" in entry:  # Eval log
            eval_losses.append(entry["eval_loss"])
            eval_steps.append(step)

    return {
        "train_loss": train_losses,
        "eval_loss": eval_losses,
        "grad_norm": grad_norms,
        "learning_rate": learning_rates,
        "step": steps,
        "epoch": epochs,
        "eval_step": eval_steps,
    }


def smooth_values(values: list, window: int = 5) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(values) < window:
        return np.array(values)

    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(values)]


def plot_loss_curve(metrics: dict, config: dict, output_path: Path):
    """Generate loss curve plot with annotations."""
    fig, ax = plt.subplots(figsize=(12, 7))

    steps = metrics["step"]
    train_loss = metrics["train_loss"]

    if not steps or not train_loss:
        print("  Warning: No training loss data to plot")
        return

    # Raw loss (faded)
    ax.plot(steps, train_loss, alpha=0.3, color='blue', linewidth=1, label='Train Loss (raw)')

    # Smoothed loss
    smoothed = smooth_values(train_loss, window=5)
    ax.plot(steps, smoothed, color='blue', linewidth=2, label='Train Loss (smoothed)')

    # Eval loss points
    if metrics["eval_loss"] and metrics["eval_step"]:
        ax.scatter(metrics["eval_step"], metrics["eval_loss"],
                   color='red', s=100, zorder=5, marker='o', label='Eval Loss')
        for step, loss in zip(metrics["eval_step"], metrics["eval_loss"]):
            ax.annotate(f'{loss:.3f}', (step, loss),
                       textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add hyperparameter annotations
    hyperparam_text = format_hyperparams(config)
    ax.text(0.02, 0.98, hyperparam_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add final metrics
    if train_loss:
        final_loss = train_loss[-1]
        min_loss = min(train_loss)
        metrics_text = f"Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}"
        if metrics["eval_loss"]:
            metrics_text += f"\nBest Eval: {min(metrics['eval_loss']):.4f}"
        ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_learning_rate(metrics: dict, config: dict, output_path: Path):
    """Generate learning rate schedule plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = metrics["step"]
    lr = metrics["learning_rate"]

    if not steps or not lr:
        print("  Warning: No learning rate data to plot")
        return

    ax.plot(steps, lr, color='green', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate max LR
    max_lr = max(lr)
    max_lr_step = steps[lr.index(max_lr)]
    ax.annotate(f'Max: {max_lr:.2e}', (max_lr_step, max_lr),
                textcoords="offset points", xytext=(10, -10), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_grad_norm(metrics: dict, config: dict, output_path: Path):
    """Generate gradient norm plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = metrics["step"]
    grad_norm = metrics["grad_norm"]

    if not steps or not grad_norm:
        print("  Warning: No gradient norm data to plot")
        return

    # Ensure same length
    min_len = min(len(steps), len(grad_norm))
    steps = steps[:min_len]
    grad_norm = grad_norm[:min_len]

    ax.plot(steps, grad_norm, alpha=0.5, color='purple', linewidth=1)
    smoothed = smooth_values(grad_norm, window=5)
    ax.plot(steps, smoothed, color='purple', linewidth=2)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add stats
    mean_norm = np.mean(grad_norm)
    max_norm = max(grad_norm)
    ax.axhline(y=mean_norm, color='orange', linestyle='--', alpha=0.7, label=f'Mean: {mean_norm:.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary(metrics: dict, config: dict, output_path: Path):
    """Generate 2x2 summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = metrics["step"]

    # Loss curve (top-left)
    ax = axes[0, 0]
    if metrics["train_loss"]:
        ax.plot(steps, metrics["train_loss"], alpha=0.3, color='blue', linewidth=1)
        smoothed = smooth_values(metrics["train_loss"], window=5)
        ax.plot(steps, smoothed, color='blue', linewidth=2, label='Train')
    if metrics["eval_loss"] and metrics["eval_step"]:
        ax.scatter(metrics["eval_step"], metrics["eval_loss"],
                   color='red', s=80, zorder=5, label='Eval')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate (top-right)
    ax = axes[0, 1]
    if metrics["learning_rate"]:
        ax.plot(steps[:len(metrics["learning_rate"])], metrics["learning_rate"],
                color='green', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    # Gradient norm (bottom-left)
    ax = axes[1, 0]
    if metrics["grad_norm"]:
        min_len = min(len(steps), len(metrics["grad_norm"]))
        ax.plot(steps[:min_len], metrics["grad_norm"][:min_len],
                alpha=0.5, color='purple', linewidth=1)
        smoothed = smooth_values(metrics["grad_norm"][:min_len], window=5)
        ax.plot(steps[:min_len], smoothed, color='purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)

    # Config & stats (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')

    config_text = format_hyperparams(config, verbose=True)
    stats_text = format_stats(metrics)
    full_text = f"Configuration:\n{config_text}\n\nTraining Stats:\n{stats_text}"

    ax.text(0.1, 0.9, full_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.suptitle(f"Training Summary: {config.get('model_short', 'Unknown')}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def format_hyperparams(config: dict, verbose: bool = False) -> str:
    """Format hyperparameters for display."""
    lines = []

    if verbose:
        if "base_model" in config:
            lines.append(f"Model: {config['base_model'].split('/')[-1]}")
        if "training_data" in config:
            lines.append(f"Data: {Path(config['training_data']).name}")

    if "lora_r" in config:
        lines.append(f"LoRA r={config['lora_r']}, α={config.get('lora_alpha', 'N/A')}")
    if "lora_dropout" in config:
        lines.append(f"Dropout: {config['lora_dropout']}")
    if "learning_rate" in config:
        lines.append(f"LR: {config['learning_rate']:.0e}")
    if "epochs" in config:
        lines.append(f"Epochs: {config['epochs']}")
    if "batch_size" in config and "gradient_accumulation_steps" in config:
        eff_batch = config["batch_size"] * config["gradient_accumulation_steps"]
        lines.append(f"Batch: {config['batch_size']}×{config['gradient_accumulation_steps']}={eff_batch}")
    if "quantization" in config:
        lines.append(f"Quant: {config['quantization']}")

    return "\n".join(lines)


def format_stats(metrics: dict) -> str:
    """Format training statistics for display."""
    lines = []

    if metrics["train_loss"]:
        lines.append(f"Final Loss: {metrics['train_loss'][-1]:.4f}")
        lines.append(f"Min Loss: {min(metrics['train_loss']):.4f}")

    if metrics["eval_loss"]:
        lines.append(f"Best Eval: {min(metrics['eval_loss']):.4f}")

    if metrics["step"]:
        lines.append(f"Total Steps: {metrics['step'][-1]}")

    if metrics["epoch"]:
        lines.append(f"Epochs: {metrics['epoch'][-1]:.2f}")

    return "\n".join(lines)


def generate_report(metrics: dict, config: dict, output_path: Path):
    """Generate text report of training."""
    lines = [
        "=" * 60,
        "Training Report",
        "=" * 60,
        "",
        "Configuration:",
        "-" * 40,
    ]

    for key, value in config.items():
        lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "Training Statistics:",
        "-" * 40,
    ])

    if metrics["train_loss"]:
        lines.append(f"  Initial Loss: {metrics['train_loss'][0]:.4f}")
        lines.append(f"  Final Loss: {metrics['train_loss'][-1]:.4f}")
        lines.append(f"  Min Loss: {min(metrics['train_loss']):.4f}")
        lines.append(f"  Loss Reduction: {(1 - metrics['train_loss'][-1]/metrics['train_loss'][0])*100:.1f}%")

    if metrics["eval_loss"]:
        lines.append(f"  Best Eval Loss: {min(metrics['eval_loss']):.4f}")
        lines.append(f"  Final Eval Loss: {metrics['eval_loss'][-1]:.4f}")

    if metrics["step"]:
        lines.append(f"  Total Steps: {metrics['step'][-1]}")

    if metrics["epoch"]:
        lines.append(f"  Epochs Completed: {metrics['epoch'][-1]:.2f}")

    lines.extend([
        "",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
    ])

    output_path.write_text("\n".join(lines))
    print(f"  Saved: {output_path}")


def plot_comparison(model_dirs: list[Path], output_path: Path):
    """Generate comparison plot for multiple training runs."""
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_dirs)))
    legends = []

    for i, model_dir in enumerate(model_dirs):
        state = find_trainer_state(model_dir)
        if not state:
            print(f"  Warning: No trainer state found for {model_dir}")
            continue

        config = load_training_config(model_dir)
        metrics = extract_metrics(state.get("log_history", []))

        if not metrics["step"] or not metrics["train_loss"]:
            continue

        # Create label from key hyperparams
        label_parts = [model_dir.name.split("-")[-1]]  # Timestamp
        if config.get("lora_r"):
            label_parts.append(f"r={config['lora_r']}")
        if config.get("learning_rate"):
            label_parts.append(f"lr={config['learning_rate']:.0e}")
        if config.get("epochs"):
            label_parts.append(f"ep={config['epochs']}")
        label = " | ".join(label_parts)

        # Plot smoothed loss
        smoothed = smooth_values(metrics["train_loss"], window=5)
        ax.plot(metrics["step"], smoothed, color=colors[i], linewidth=2, label=label)

        # Add eval points
        if metrics["eval_loss"] and metrics["eval_step"]:
            ax.scatter(metrics["eval_step"], metrics["eval_loss"],
                       color=colors[i], s=60, alpha=0.7, zorder=5)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison: {output_path}")


def process_model_dir(model_dir: Path, output_subdir: str = "Parameter_Tuning"):
    """Process a single model directory and generate all plots."""
    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"Error: Directory not found: {model_dir}")
        return False

    print(f"\nProcessing: {model_dir.name}")

    # Load data
    state = find_trainer_state(model_dir)
    if not state:
        print("  Error: No trainer_state.json or training_metrics.json found")
        return False

    config = load_training_config(model_dir)
    metrics = extract_metrics(state.get("log_history", []))

    if not metrics["step"]:
        print("  Error: No training data found in log history")
        return False

    print(f"  Found {len(metrics['step'])} logged steps")

    # Create output directory
    output_dir = model_dir / output_subdir
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_loss_curve(metrics, config, output_dir / "loss_curve.png")
    plot_learning_rate(metrics, config, output_dir / "learning_rate.png")
    plot_grad_norm(metrics, config, output_dir / "grad_norm.png")
    plot_summary(metrics, config, output_dir / "training_summary.png")
    generate_report(metrics, config, output_dir / "training_report.txt")

    print(f"  Output: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from HuggingFace checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s models/gswa-lora-Mistral-20260130-*/
  %(prog)s --compare models/gswa-lora-Mistral-*/
  %(prog)s models/gswa-lora-* --output-dir analysis/
        """
    )
    parser.add_argument("model_dirs", nargs="+", type=Path,
                        help="Model directory or directories to process")
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison plot instead of individual plots")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Custom output directory (default: {model_dir}/Parameter_Tuning/)")
    args = parser.parse_args()

    # Expand glob patterns
    model_dirs = []
    for path in args.model_dirs:
        if "*" in str(path):
            model_dirs.extend(Path(".").glob(str(path)))
        elif path.is_dir():
            model_dirs.append(path)

    if not model_dirs:
        print("Error: No valid model directories found")
        return 1

    model_dirs = sorted(set(model_dirs))
    print(f"Found {len(model_dirs)} model director{'ies' if len(model_dirs) > 1 else 'y'}")

    if args.compare and len(model_dirs) > 1:
        # Comparison mode
        output_path = args.output_dir or Path("comparison_plot.png")
        plot_comparison(model_dirs, output_path)
    else:
        # Individual plots
        for model_dir in model_dirs:
            output_subdir = str(args.output_dir) if args.output_dir else "Parameter_Tuning"
            process_model_dir(model_dir, output_subdir)

    return 0


if __name__ == "__main__":
    exit(main())
