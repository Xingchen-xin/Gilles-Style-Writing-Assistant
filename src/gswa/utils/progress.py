"""
Progress bar and status display utilities.
Supports real-time visualization of training progress.
"""
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta


class Colors:
    """Terminal color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports color."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Add color if terminal supports it, otherwise return plain text."""
        if cls.supports_color():
            return f"{color}{text}{cls.ENDC}"
        return text


@dataclass
class ProgressState:
    """Progress state tracking."""
    current: int
    total: int
    description: str = ""
    start_time: Optional[datetime] = None

    @property
    def percentage(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0

    @property
    def elapsed(self) -> timedelta:
        if self.start_time:
            return datetime.now() - self.start_time
        return timedelta(0)

    @property
    def eta(self) -> Optional[timedelta]:
        if self.current > 0 and self.start_time:
            elapsed = self.elapsed.total_seconds()
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate
            return timedelta(seconds=remaining)
        return None


class ProgressBar:
    """
    Real-time progress bar.

    Usage:
        with ProgressBar(total=100, desc="Training") as pbar:
            for i in range(100):
                # do work
                pbar.update(1)
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 40,
        show_eta: bool = True,
        show_speed: bool = True,
    ):
        self.total = total
        self.desc = desc
        self.width = width
        self.show_eta = show_eta
        self.show_speed = show_speed
        self.current = 0
        self.start_time: Optional[datetime] = None
        self._last_update = 0.0

    def __enter__(self):
        self.start_time = datetime.now()
        self._render()
        return self

    def __exit__(self, *args):
        self._render(final=True)
        print()  # newline

    def update(self, n: int = 1):
        """Update progress."""
        self.current = min(self.current + n, self.total)

        # Rate limit refresh (max every 0.1s)
        now = time.time()
        if now - self._last_update >= 0.1 or self.current >= self.total:
            self._render()
            self._last_update = now

    def set_description(self, desc: str):
        """Update description."""
        self.desc = desc
        self._render()

    def _render(self, final: bool = False):
        """Render progress bar."""
        percentage = (self.current / self.total * 100) if self.total > 0 else 0
        filled = int(self.width * self.current / self.total) if self.total > 0 else 0

        # Progress bar characters
        bar = '\u2588' * filled + '\u2591' * (self.width - filled)

        # Colors based on progress
        if percentage >= 100:
            bar_color = Colors.GREEN
            pct_color = Colors.GREEN
        elif percentage >= 50:
            bar_color = Colors.CYAN
            pct_color = Colors.CYAN
        else:
            bar_color = Colors.BLUE
            pct_color = Colors.YELLOW

        # Build display string
        parts = []

        # Description
        if self.desc:
            parts.append(f"{Colors.colorize(self.desc, Colors.BOLD)}")

        # Progress bar
        parts.append(f"|{Colors.colorize(bar, bar_color)}|")

        # Percentage
        parts.append(Colors.colorize(f"{percentage:5.1f}%", pct_color))

        # Count
        parts.append(f"[{self.current}/{self.total}]")

        # Speed and ETA
        if self.start_time and self.current > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            speed = self.current / elapsed if elapsed > 0 else 0

            if self.show_speed:
                parts.append(f"{speed:.1f}it/s")

            if self.show_eta and self.current < self.total:
                remaining = (self.total - self.current) / speed if speed > 0 else 0
                eta_str = self._format_time(remaining)
                parts.append(f"ETA:{eta_str}")
            elif final:
                parts.append(f"Time:{self._format_time(elapsed)}")

        # Output
        line = " ".join(parts)
        sys.stdout.write(f"\r{line}" + " " * 10)  # Extra space clears residual chars
        sys.stdout.flush()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class StepProgress:
    """
    Multi-step progress display.

    Usage:
        steps = StepProgress([
            "Detect corpus",
            "Prepare data",
            "Load model",
            "Start training",
            "Save model"
        ])

        steps.start(0)
        # do step 0
        steps.complete(0)

        steps.start(1)
        # do step 1
        steps.complete(1)
    """

    def __init__(self, steps: List[str]):
        self.steps = steps
        self.status = ["pending"] * len(steps)  # pending, running, done, failed
        self.start_times: Dict[int, datetime] = {}

    def start(self, index: int):
        """Start a step."""
        self.status[index] = "running"
        self.start_times[index] = datetime.now()
        self._render()

    def complete(self, index: int, message: str = ""):
        """Complete a step."""
        self.status[index] = "done"
        self._render()
        if message:
            print(f"   {Colors.colorize('->', Colors.DIM)} {message}")

    def fail(self, index: int, error: str = ""):
        """Mark step as failed."""
        self.status[index] = "failed"
        self._render()
        if error:
            print(f"   {Colors.colorize('x', Colors.RED)} {error}")

    def _render(self):
        """Render all steps."""
        print()
        print(Colors.colorize("=" * 60, Colors.DIM))

        for i, step in enumerate(self.steps):
            status = self.status[i]

            if status == "pending":
                icon = Colors.colorize("o", Colors.DIM)
                text = Colors.colorize(step, Colors.DIM)
            elif status == "running":
                icon = Colors.colorize("*", Colors.YELLOW)
                text = Colors.colorize(step, Colors.YELLOW)
                elapsed = ""
                if i in self.start_times:
                    secs = (datetime.now() - self.start_times[i]).total_seconds()
                    elapsed = f" ({secs:.1f}s)"
                text += Colors.colorize(elapsed, Colors.DIM)
            elif status == "done":
                icon = Colors.colorize("+", Colors.GREEN)
                text = Colors.colorize(step, Colors.GREEN)
            else:  # failed
                icon = Colors.colorize("x", Colors.RED)
                text = Colors.colorize(step, Colors.RED)

            print(f"  {icon} Step {i+1}/{len(self.steps)}: {text}")

        print(Colors.colorize("=" * 60, Colors.DIM))


def print_header(title: str, width: int = 60):
    """Print title header."""
    print()
    print(Colors.colorize("=" * width, Colors.CYAN))
    padding = (width - len(title) - 4) // 2
    print(Colors.colorize("=" * padding + f"  {title}  " + "=" * padding, Colors.CYAN))
    print(Colors.colorize("=" * width, Colors.CYAN))


def print_section(title: str, width: int = 60):
    """Print section title."""
    print()
    line_len = width - len(title) - 5
    print(Colors.colorize(f"--- {title} " + "-" * line_len, Colors.BLUE))


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.colorize('[OK]', Colors.GREEN)} {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.colorize('[WARN]', Colors.YELLOW)} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.colorize('[ERROR]', Colors.RED)} {message}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.colorize('[INFO]', Colors.BLUE)} {message}")


def confirm(prompt: str, default: bool = False) -> bool:
    """Confirmation prompt."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{prompt} {suffix}: ").strip().lower()

    if not response:
        return default
    return response in ('y', 'yes')


def select_option(prompt: str, options: List[str], default: int = 0) -> int:
    """Option selection."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = "->" if i == default else "  "
        print(f"  {marker} [{i+1}] {opt}")

    while True:
        response = input(f"Select [1-{len(options)}] (default {default+1}): ").strip()
        if not response:
            return default
        try:
            idx = int(response) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}")
