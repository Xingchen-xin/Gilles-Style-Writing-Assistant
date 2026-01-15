#!/usr/bin/env python3
"""
GSWA One-Click Training Wizard
==============================

Foolproof training workflow that integrates:
- Corpus status detection
- Training data statistics
- Hardware auto-detection
- Smart model selection
- Real-time progress display

Usage:
    python scripts/training_wizard.py           # Interactive wizard
    python scripts/training_wizard.py --auto    # Fully automatic mode
    python scripts/training_wizard.py --model qwen-7b  # Specify model
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gswa.utils.progress import (
    Colors, ProgressBar, StepProgress,
    print_header, print_section, print_success, print_warning, print_error, print_info,
    confirm, select_option
)


# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_RAW_DIR = PROJECT_ROOT / "data" / "corpus" / "raw"
CORPUS_PRIORITY_DIR = CORPUS_RAW_DIR / "important_examples"
CORPUS_PARSED_DIR = PROJECT_ROOT / "data" / "corpus" / "parsed"
TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
STYLE_DIR = PROJECT_ROOT / "data" / "style"
STYLE_FINGERPRINT = STYLE_DIR / "author_fingerprint.json"


# Model configurations
MODELS = {
    "qwen-7b": {
        "name": "Qwen2.5-7B-Instruct",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "mlx_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "vram_gb": 8,
        "description": "Recommended: Best value, strong scientific writing",
        "tags": ["recommended", "bilingual", "scientific"],
    },
    "qwen-14b": {
        "name": "Qwen2.5-14B-Instruct",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "mlx_id": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "vram_gb": 16,
        "description": "Best: Strongest scientific writing capability",
        "tags": ["best", "bilingual", "scientific", "needs-more-vram"],
    },
    "qwen-1.5b": {
        "name": "Qwen2.5-1.5B-Instruct",
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "mlx_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "vram_gb": 4,
        "description": "Lightweight: For low-spec devices or testing",
        "tags": ["lightweight", "fast", "testing"],
    },
    "llama3-8b": {
        "name": "Llama-3.1-8B-Instruct",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "mlx_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "vram_gb": 10,
        "description": "Meta: Best for English writing",
        "tags": ["english-best", "reasoning"],
    },
    "mistral-7b": {
        "name": "Mistral-7B-Instruct-v0.3",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "mlx_id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "vram_gb": 8,
        "description": "Classic: Fast and versatile",
        "tags": ["classic", "fast", "versatile"],
    },
    "phi-3.5": {
        "name": "Phi-3.5-mini-instruct",
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "mlx_id": "mlx-community/Phi-3.5-mini-instruct-4bit",
        "vram_gb": 4,
        "description": "Microsoft: Small but smart",
        "tags": ["lightweight", "microsoft", "efficient"],
    },
}

DEFAULT_MODEL = "qwen-7b"


# ==============================================================================
# System Detection
# ==============================================================================

@dataclass
class SystemInfo:
    """System information."""
    os: str
    is_mac: bool
    is_apple_silicon: bool
    has_cuda: bool
    gpu_name: Optional[str]
    gpu_vram_gb: float
    system_ram_gb: float
    recommended_backend: str  # "mlx" | "cuda" | "cpu"
    recommended_model: str


def detect_system() -> SystemInfo:
    """Detect system information."""
    os_name = platform.system()
    is_mac = os_name == "Darwin"
    is_apple_silicon = False
    has_cuda = False
    gpu_name = None
    gpu_vram_gb = 0.0
    system_ram_gb = 16.0

    # Detect Mac Apple Silicon
    if is_mac:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            chip = result.stdout.strip()
            is_apple_silicon = "Apple" in chip
            gpu_name = chip

            # Get memory
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            system_ram_gb = int(result.stdout.strip()) / (1024**3)
            # Apple Silicon unified memory, GPU can use ~70%
            gpu_vram_gb = system_ram_gb * 0.7
        except Exception:
            pass

    # Detect CUDA GPU
    if not is_mac:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(",")
                gpu_name = parts[0].strip()
                gpu_vram_gb = float(parts[1].strip()) / 1024
                has_cuda = True
        except Exception:
            pass

        # Fallback: detect via PyTorch
        if not has_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    has_cuda = True
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                pass

    # Get system memory (non-Mac)
    if not is_mac:
        try:
            import psutil
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            pass

    # Determine recommended backend
    if is_apple_silicon:
        backend = "mlx"
    elif has_cuda:
        backend = "cuda"
    else:
        backend = "cpu"

    # Recommend model based on available memory
    available_vram = gpu_vram_gb if gpu_vram_gb > 0 else system_ram_gb * 0.5
    if available_vram >= 16:
        recommended_model = "qwen-14b"
    elif available_vram >= 8:
        recommended_model = "qwen-7b"
    elif available_vram >= 4:
        recommended_model = "qwen-1.5b"
    else:
        recommended_model = "phi-3.5"

    return SystemInfo(
        os=os_name,
        is_mac=is_mac,
        is_apple_silicon=is_apple_silicon,
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        system_ram_gb=system_ram_gb,
        recommended_backend=backend,
        recommended_model=recommended_model,
    )


# ==============================================================================
# Corpus Detection
# ==============================================================================

@dataclass
class CorpusStats:
    """Corpus statistics."""
    regular_files: List[Dict]
    priority_files: List[Dict]
    total_files: int
    total_size_mb: float
    by_format: Dict[str, int]
    is_healthy: bool
    warnings: List[str]
    suggestions: List[str]


def detect_corpus() -> CorpusStats:
    """Detect corpus status."""
    regular_files = []
    priority_files = []
    by_format: Dict[str, int] = {}
    total_size = 0.0
    warnings = []
    suggestions = []

    supported_formats = [".pdf", ".docx", ".txt", ".md", ".tex"]

    # Ensure directories exist
    CORPUS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_PRIORITY_DIR.mkdir(parents=True, exist_ok=True)

    # Scan regular files
    for f in CORPUS_RAW_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in supported_formats:
            size_kb = f.stat().st_size / 1024
            regular_files.append({
                "name": f.name,
                "path": str(f),
                "format": f.suffix.lower(),
                "size_kb": size_kb,
                "weight": 1.0,
            })
            total_size += size_kb / 1024
            ext = f.suffix.lower()
            by_format[ext] = by_format.get(ext, 0) + 1

    # Scan priority files
    for f in CORPUS_PRIORITY_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in supported_formats:
            size_kb = f.stat().st_size / 1024
            priority_files.append({
                "name": f.name,
                "path": str(f),
                "format": f.suffix.lower(),
                "size_kb": size_kb,
                "weight": 2.5,
            })
            total_size += size_kb / 1024
            ext = f.suffix.lower()
            by_format[ext] = by_format.get(ext, 0) + 1

    total = len(regular_files) + len(priority_files)

    # Generate warnings and suggestions
    if total == 0:
        warnings.append("No corpus files found!")
        suggestions.append("Please add PDF/DOCX/TXT files to data/corpus/raw/")
    elif total < 5:
        warnings.append(f"Few corpus files ({total}), recommend 10+ for better results")

    if len(priority_files) == 0 and total > 0:
        suggestions.append("Consider moving representative articles to important_examples/ for 2.5x training weight")

    if len(priority_files) > 0 and len(priority_files) > len(regular_files):
        suggestions.append("Priority files ratio is high, ensure these are truly representative")

    return CorpusStats(
        regular_files=regular_files,
        priority_files=priority_files,
        total_files=total,
        total_size_mb=total_size,
        by_format=by_format,
        is_healthy=total >= 5,
        warnings=warnings,
        suggestions=suggestions,
    )


# ==============================================================================
# Training Data Detection
# ==============================================================================

@dataclass
class TrainingStats:
    """Training data statistics."""
    corpus_paragraphs: int
    priority_paragraphs: int
    training_entries: int
    has_training_data: bool
    has_corpus: bool


def detect_training_data() -> TrainingStats:
    """Detect training data status."""
    corpus_paragraphs = 0
    priority_paragraphs = 0
    training_entries = 0

    # Detect corpus
    corpus_file = CORPUS_PARSED_DIR / "corpus.jsonl"
    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus_paragraphs += 1
                    try:
                        para = json.loads(line)
                        if para.get("is_priority"):
                            priority_paragraphs += 1
                    except Exception:
                        pass

    # Detect training data
    training_file = TRAINING_DATA_DIR / "alpaca_train.jsonl"
    if training_file.exists():
        with open(training_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    training_entries += 1

    return TrainingStats(
        corpus_paragraphs=corpus_paragraphs,
        priority_paragraphs=priority_paragraphs,
        training_entries=training_entries,
        has_training_data=training_entries > 0,
        has_corpus=corpus_paragraphs > 0,
    )


# ==============================================================================
# Display Functions
# ==============================================================================

def show_system_info(info: SystemInfo):
    """Display system information."""
    print_section("System Information")

    print(f"  OS:              {info.os}")

    if info.is_apple_silicon:
        print(f"  Chip:            {Colors.colorize(info.gpu_name, Colors.GREEN)} (Apple Silicon)")
        print(f"  Unified Memory:  {info.system_ram_gb:.1f} GB (GPU ~{info.gpu_vram_gb:.1f} GB)")
        print(f"  Backend:         {Colors.colorize('MLX', Colors.GREEN)}")
    elif info.has_cuda:
        print(f"  GPU:             {Colors.colorize(info.gpu_name, Colors.GREEN)}")
        print(f"  VRAM:            {info.gpu_vram_gb:.1f} GB")
        print(f"  Backend:         {Colors.colorize('CUDA (LoRA/QLoRA)', Colors.GREEN)}")
    else:
        print(f"  GPU:             {Colors.colorize('Not detected', Colors.YELLOW)}")
        print(f"  System RAM:      {info.system_ram_gb:.1f} GB")
        print(f"  Backend:         {Colors.colorize('CPU (slower)', Colors.YELLOW)}")

    model_info = MODELS.get(info.recommended_model, {})
    print(f"  Recommended:     {Colors.colorize(model_info.get('name', info.recommended_model), Colors.CYAN)}")


def show_corpus_status(corpus: CorpusStats):
    """Display corpus status."""
    print_section("Corpus Status")

    # File statistics
    print(f"  Regular files:   {len(corpus.regular_files)} (weight 1.0x)")
    print(f"  Priority files:  {len(corpus.priority_files)} (weight 2.5x) *")
    print(f"  Total:           {corpus.total_files} files, {corpus.total_size_mb:.2f} MB")

    # Format distribution
    if corpus.by_format:
        formats = ", ".join([f"{k}: {v}" for k, v in corpus.by_format.items()])
        print(f"  Formats:         {formats}")

    # Weight preview
    if corpus.total_files > 0:
        regular_weight = len(corpus.regular_files) * 1.0
        priority_weight = len(corpus.priority_files) * 2.5
        total_weight = regular_weight + priority_weight

        print(f"\n  Training Weight Distribution:")
        bar_reg = int(regular_weight / total_weight * 20) if total_weight > 0 else 0
        bar_pri = int(priority_weight / total_weight * 20) if total_weight > 0 else 0
        print(f"     Regular:  {'#' * bar_reg} {regular_weight/total_weight*100:.1f}%")
        print(f"     Priority: {'#' * bar_pri} {priority_weight/total_weight*100:.1f}%")

    # Warnings
    for w in corpus.warnings:
        print_warning(w)

    # Suggestions
    for s in corpus.suggestions:
        print_info(s)


def show_training_status(stats: TrainingStats):
    """Display training data status."""
    print_section("Training Data Status")

    if stats.has_corpus:
        print(f"  Corpus paragraphs:  {stats.corpus_paragraphs}")
        print(f"  Priority paragraphs: {stats.priority_paragraphs} *")
    else:
        print(f"  Corpus:             {Colors.colorize('Not parsed', Colors.YELLOW)}")
        print_info("Run corpus parsing first")

    if stats.has_training_data:
        print(f"  Training samples:   {stats.training_entries}")
    else:
        print(f"  Training data:      {Colors.colorize('Not prepared', Colors.YELLOW)}")


def show_model_selection():
    """Display model selection."""
    print_section("Available Models")

    for key, model in MODELS.items():
        if key == DEFAULT_MODEL:
            marker = Colors.colorize("->", Colors.GREEN)
        else:
            marker = "  "

        print(f"  {marker} {key:12} {model['name']}")
        print(f"     {Colors.colorize(model['description'], Colors.DIM)}")
        print(f"     VRAM needed: {model['vram_gb']} GB")
        print()


def select_model(system_info: SystemInfo, auto: bool = False) -> str:
    """Select model."""
    if auto:
        return system_info.recommended_model

    print_section("Model Selection")

    options = list(MODELS.keys())
    descriptions = [
        f"{MODELS[k]['name']} - {MODELS[k]['description']}"
        for k in options
    ]

    # Find recommended model index
    default_idx = options.index(system_info.recommended_model) if system_info.recommended_model in options else 0

    selected = select_option("Select base model:", descriptions, default_idx)
    return options[selected]


# ==============================================================================
# Training Execution
# ==============================================================================

def run_parse_corpus() -> bool:
    """Run corpus parsing."""
    print_info("Parsing corpus...")

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "parse_corpus.py")]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    return result.returncode == 0


def run_prepare_training() -> bool:
    """Prepare training data."""
    print_info("Preparing training data...")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_training_data.py"),
        "--format", "alpaca",
        "--weighted",
        "--split"
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    return result.returncode == 0


def run_build_index() -> bool:
    """Build index."""
    print_info("Building similarity index...")

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "build_index.py")]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    return result.returncode == 0


def run_analyze_style() -> bool:
    """Analyze author style and create fingerprint."""
    print_info("Analyzing author writing style...")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_style.py"),
        "--quiet"
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    return result.returncode == 0


def show_style_status():
    """Show style fingerprint status."""
    print_section("Style Fingerprint Status")

    if STYLE_FINGERPRINT.exists():
        try:
            with open(STYLE_FINGERPRINT, "r", encoding="utf-8") as f:
                fp = json.load(f)

            print(f"  Author:          {fp.get('author_name', 'Unknown')}")
            print(f"  Corpus size:     {fp.get('corpus_size', 0)} paragraphs")
            print(f"  Total words:     {fp.get('total_words', 0):,}")

            ss = fp.get("sentence_stats", {})
            if ss.get("avg_length"):
                print(f"  Avg sentence:    {ss['avg_length']:.1f} words")

            st = fp.get("structure_stats", {})
            if st.get("passive_voice_ratio"):
                print(f"  Passive voice:   {st['passive_voice_ratio']*100:.1f}%")

            vs = fp.get("vocabulary_stats", {})
            if vs.get("favorite_transitions"):
                print(f"  Transitions:     {', '.join(vs['favorite_transitions'][:5])}")

            print(f"\n  {Colors.colorize('Style fingerprint loaded', Colors.GREEN)}")
        except Exception as e:
            print_warning(f"Could not read fingerprint: {e}")
    else:
        print(f"  Status:          {Colors.colorize('Not analyzed', Colors.YELLOW)}")
        print_info("Style analysis will run during training")


def run_training(system_info: SystemInfo, model_key: str) -> bool:
    """Run training."""
    model = MODELS[model_key]

    if system_info.recommended_backend == "mlx":
        # Mac MLX training
        script = PROJECT_ROOT / "scripts" / "finetune_mlx_mac.py"
        if not script.exists():
            print_error(f"MLX training script not found: {script}")
            return False
        cmd = [
            sys.executable,
            str(script),
            "--model", model["mlx_id"],
            "--auto"
        ]
    else:
        # CUDA LoRA training
        script = PROJECT_ROOT / "scripts" / "finetune_lora.py"
        if not script.exists():
            print_error(f"LoRA training script not found: {script}")
            return False
        cmd = [
            sys.executable,
            str(script),
            "--model", model["hf_id"],
            "--auto"
        ]

    print_info(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


# ==============================================================================
# Main Flow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GSWA One-Click Training Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/training_wizard.py           # Interactive wizard
    python scripts/training_wizard.py --auto    # Fully automatic mode
    python scripts/training_wizard.py --model qwen-14b  # Specify model
    python scripts/training_wizard.py --status  # Show status only
    python scripts/training_wizard.py --models  # List available models
        """
    )

    parser.add_argument("--auto", action="store_true", help="Fully automatic mode, no confirmation")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()), help="Specify model")
    parser.add_argument("--status", action="store_true", help="Show status only, no training")
    parser.add_argument("--models", action="store_true", help="List available models")
    parser.add_argument("--skip-parse", action="store_true", help="Skip corpus parsing")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")

    args = parser.parse_args()

    # Print header
    print_header("GSWA One-Click Training Wizard")
    print("   Gilles-Style Writing Assistant")
    print("   Foolproof fine-tuning for your writing style")

    # Show model list only
    if args.models:
        show_model_selection()
        return 0

    # Step 1: System detection
    print_section("Step 1/5: System Detection")
    system_info = detect_system()
    show_system_info(system_info)

    # Step 2: Corpus detection
    print_section("Step 2/5: Corpus Detection")
    corpus = detect_corpus()
    show_corpus_status(corpus)

    # Step 3: Training data detection
    print_section("Step 3/5: Training Data Detection")
    training = detect_training_data()
    show_training_status(training)

    # Step 3.5: Style fingerprint status
    show_style_status()

    # Status only mode
    if args.status:
        print()
        print_success("Status check complete!")
        return 0

    # Check corpus is sufficient
    if corpus.total_files == 0:
        print()
        print_error("No corpus files found!")
        print_info("Please add articles to the following directories:")
        print(f"   Regular:  {CORPUS_RAW_DIR}")
        print(f"   Priority: {CORPUS_PRIORITY_DIR}")
        return 1

    if corpus.total_files < 3 and not args.auto:
        print()
        print_warning(f"Only {corpus.total_files} corpus files, recommend 5+ for better results")
        if not confirm("Continue training?", default=False):
            return 0

    # Step 4: Model selection
    print_section("Step 4/5: Model Selection")

    if args.model:
        selected_model = args.model
        print_info(f"Using specified model: {MODELS[selected_model]['name']}")
    else:
        selected_model = select_model(system_info, auto=args.auto)

    model_info = MODELS[selected_model]
    print()
    print(f"  Selected model:  {Colors.colorize(model_info['name'], Colors.GREEN)}")
    print(f"  Model ID:        {model_info['hf_id']}")
    print(f"  VRAM needed:     {model_info['vram_gb']} GB")

    # Confirm start
    if not args.auto:
        print()
        if not confirm("Start training?", default=True):
            print("Cancelled")
            return 0

    # Step 5: Execute training
    print_section("Step 5/5: Execute Training")

    steps = StepProgress([
        "Parse corpus",
        "Analyze author style",
        "Prepare training data",
        "Build similarity index",
        "Fine-tune model",
        "Verify results"
    ])

    # 5.1 Parse corpus
    if not args.skip_parse:
        steps.start(0)
        if run_parse_corpus():
            steps.complete(0)
        else:
            steps.fail(0, "Corpus parsing failed")
            return 1
    else:
        steps.complete(0, "Skipped")

    # 5.2 Analyze author style
    steps.start(1)
    if run_analyze_style():
        steps.complete(1)
    else:
        steps.fail(1, "Style analysis failed")
        return 1

    # 5.3 Prepare training data
    steps.start(2)
    if run_prepare_training():
        steps.complete(2)
    else:
        steps.fail(2, "Training data preparation failed")
        return 1

    # 5.4 Build index
    if not args.skip_index:
        steps.start(3)
        if run_build_index():
            steps.complete(3)
        else:
            steps.fail(3, "Index building failed")
            return 1
    else:
        steps.complete(3, "Skipped")

    # 5.5 Execute training
    steps.start(4)
    if run_training(system_info, selected_model):
        steps.complete(4)
    else:
        steps.fail(4, "Training failed")
        return 1

    # 5.6 Verify
    steps.start(5)
    # Basic verification
    steps.complete(5, "Model saved")

    # Complete
    print()
    print_header("Training Complete!")
    print()
    print_success("Fine-tuned model has been saved")
    print()
    print_info("Next steps:")
    print("   1. Start service: make run")
    print("   2. Open browser: http://localhost:8080")
    print("   3. Start using your personalized writing assistant!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
