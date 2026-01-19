"""
Training Planner Module - Finds optimal training configuration.

Provides:
- Candidate plan generation
- Dry-run testing (short runs to estimate throughput/memory)
- Plan scoring based on quality and constraints
- Deterministic plan selection
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import random


@dataclass
class PlanCandidate:
    """A candidate training configuration to evaluate."""
    # Configuration
    batch_size: int = 1
    max_seq_length: int = 1024
    num_layers: int = 8
    grad_accum_steps: int = 1
    eval_batch_size: int = 1
    lora_rank: int = 8

    # Estimated metrics (filled during dry-run)
    estimated_tokens_per_sec: float = 0.0
    estimated_iters_per_sec: float = 0.0
    estimated_peak_memory_gb: float = 0.0
    dry_run_success: bool = False
    dry_run_error: str = ""

    # Computed scores
    stability_score: float = 1.0  # 0-1, based on memory margin
    throughput_score: float = 0.0  # tokens/sec normalized
    effective_batch_size: int = 0  # batch_size * grad_accum_steps
    final_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"batch={self.batch_size}, seq_len={self.max_seq_length}, "
            f"layers={self.num_layers}, grad_accum={self.grad_accum_steps}, "
            f"eval_batch={self.eval_batch_size}, lora_rank={self.lora_rank}"
        )


@dataclass
class PlannerResult:
    """Results from the training planner."""
    # Best plan
    best_plan: Optional[PlanCandidate] = None

    # All candidates evaluated
    candidates: List[PlanCandidate] = field(default_factory=list)

    # Hardware constraints
    available_memory_gb: float = 0.0
    memory_margin: float = 0.8  # Use up to 80% of available

    # Timing
    planner_started_at: str = ""
    planner_ended_at: str = ""
    total_dry_run_time_sec: float = 0.0

    # Configuration
    dry_run_steps: int = 10
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_plan": self.best_plan.to_dict() if self.best_plan else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "available_memory_gb": self.available_memory_gb,
            "memory_margin": self.memory_margin,
            "planner_started_at": self.planner_started_at,
            "planner_ended_at": self.planner_ended_at,
            "total_dry_run_time_sec": self.total_dry_run_time_sec,
            "dry_run_steps": self.dry_run_steps,
            "seed": self.seed,
        }

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class TrainingPlanner:
    """Finds optimal training configuration through dry-runs."""

    # Default candidate variations
    DEFAULT_SEQ_LENGTHS = [1024, 1536, 2048]
    DEFAULT_BATCH_SIZES = [1, 2, 4]
    DEFAULT_GRAD_ACCUM = [1, 2, 4, 8]
    DEFAULT_LORA_RANKS = [8, 16]

    def __init__(
        self,
        model_id: str,
        training_data: str,
        available_memory_gb: float,
        memory_margin: float = 0.8,
        dry_run_steps: int = 10,
        seed: int = 42,
    ):
        """Initialize planner.

        Args:
            model_id: Model to use for dry-runs
            training_data: Path to training data
            available_memory_gb: Available GPU/unified memory
            memory_margin: Safety margin (0.8 = use up to 80%)
            dry_run_steps: Steps for each dry-run
            seed: Random seed for reproducibility
        """
        self.model_id = model_id
        self.training_data = training_data
        self.available_memory_gb = available_memory_gb
        self.memory_margin = memory_margin
        self.dry_run_steps = dry_run_steps
        self.seed = seed

        # Set seeds for reproducibility
        random.seed(seed)

        self.result = PlannerResult(
            available_memory_gb=available_memory_gb,
            memory_margin=memory_margin,
            dry_run_steps=dry_run_steps,
            seed=seed,
        )

    def generate_candidates(
        self,
        seq_lengths: List[int] = None,
        batch_sizes: List[int] = None,
        grad_accum: List[int] = None,
        lora_ranks: List[int] = None,
    ) -> List[PlanCandidate]:
        """Generate candidate plans based on available memory.

        Args:
            seq_lengths: Sequence length options
            batch_sizes: Batch size options
            grad_accum: Gradient accumulation options
            lora_ranks: LoRA rank options

        Returns:
            List of feasible candidates
        """
        seq_lengths = seq_lengths or self.DEFAULT_SEQ_LENGTHS
        batch_sizes = batch_sizes or self.DEFAULT_BATCH_SIZES
        grad_accum = grad_accum or self.DEFAULT_GRAD_ACCUM
        lora_ranks = lora_ranks or self.DEFAULT_LORA_RANKS

        max_memory = self.available_memory_gb * self.memory_margin
        candidates = []

        # Generate combinations, filtering by rough memory estimate
        for seq_len in seq_lengths:
            for batch in batch_sizes:
                for accum in grad_accum:
                    for rank in lora_ranks:
                        # Rough memory estimate (very conservative)
                        # Based on: model base + batch * seq_len * embedding_dim * 4
                        estimated_mem = self._estimate_memory(batch, seq_len, rank)

                        if estimated_mem > max_memory:
                            continue

                        # Calculate num_layers based on memory
                        num_layers = self._estimate_safe_layers(
                            batch, seq_len, rank, max_memory
                        )

                        candidate = PlanCandidate(
                            batch_size=batch,
                            max_seq_length=seq_len,
                            num_layers=num_layers,
                            grad_accum_steps=accum,
                            eval_batch_size=max(1, batch // 2),
                            lora_rank=rank,
                            effective_batch_size=batch * accum,
                        )
                        candidates.append(candidate)

        # Remove duplicates
        seen = set()
        unique = []
        for c in candidates:
            key = (c.batch_size, c.max_seq_length, c.num_layers, c.grad_accum_steps, c.lora_rank)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # Sort by effective batch size (larger = better quality) then by memory
        unique.sort(
            key=lambda c: (c.effective_batch_size, c.max_seq_length),
            reverse=True
        )

        return unique

    def _estimate_memory(self, batch: int, seq_len: int, rank: int) -> float:
        """Rough memory estimate in GB."""
        # Very rough estimate based on 7B model
        # Base model: ~4GB for 4-bit
        # Activations: batch * seq_len * hidden_dim * layers * bytes
        base_gb = 4.0

        # Activation memory (rough)
        hidden_dim = 4096  # Typical for 7B
        num_layers = 32
        bytes_per_element = 2  # FP16

        activation_bytes = batch * seq_len * hidden_dim * num_layers * bytes_per_element
        activation_gb = activation_bytes / (1024 ** 3)

        # LoRA overhead
        lora_gb = rank * hidden_dim * num_layers * 2 * bytes_per_element / (1024 ** 3)

        return base_gb + activation_gb + lora_gb

    def _estimate_safe_layers(
        self,
        batch: int,
        seq_len: int,
        rank: int,
        max_memory: float
    ) -> int:
        """Estimate safe number of LoRA layers."""
        # Start conservative and increase
        for layers in [4, 8, 12, 16, 24, 32]:
            estimated = self._estimate_memory(batch, seq_len, rank) * (layers / 32)
            if estimated > max_memory * 0.7:  # Leave extra margin
                return max(4, layers - 4)
        return 32

    def run_dry_run(self, candidate: PlanCandidate) -> PlanCandidate:
        """Run a short dry-run to measure actual metrics.

        Args:
            candidate: Plan to test

        Returns:
            Updated candidate with metrics
        """
        print(f"\n  Testing: {candidate}")

        try:
            # Create a minimal MLX dry-run
            start_time = time.time()

            # Use mlx_lm lora command with minimal steps
            cmd = [
                sys.executable, "-m", "mlx_lm", "lora",
                "--model", self.model_id,
                "--train",
                "--data", str(Path(self.training_data).parent),
                "--batch-size", str(candidate.batch_size),
                "--num-layers", str(candidate.num_layers),
                "--iters", str(self.dry_run_steps),
                "--max-seq-length", str(candidate.max_seq_length),
                "--adapter-path", f"/tmp/gswa_dryrun_{self.seed}",
            ]

            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout per dry-run
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                candidate.dry_run_success = True

                # Parse output for metrics
                output = result.stdout + result.stderr

                # Try to extract tokens/sec from output
                # MLX outputs like "Iter 10: ... tok/s: 123.4"
                import re
                tok_match = re.search(r'tok/s:\s*([\d.]+)', output)
                if tok_match:
                    candidate.estimated_tokens_per_sec = float(tok_match.group(1))

                # Estimate iterations per second
                candidate.estimated_iters_per_sec = self.dry_run_steps / elapsed

                # Estimate peak memory (from MLX metal output if available)
                mem_match = re.search(r'Peak memory:\s*([\d.]+)\s*GB', output)
                if mem_match:
                    candidate.estimated_peak_memory_gb = float(mem_match.group(1))
                else:
                    # Estimate from model and config
                    candidate.estimated_peak_memory_gb = self._estimate_memory(
                        candidate.batch_size,
                        candidate.max_seq_length,
                        candidate.lora_rank
                    )

                print(f"    Success: {candidate.estimated_tokens_per_sec:.1f} tok/s, "
                      f"~{candidate.estimated_peak_memory_gb:.1f}GB peak")

            else:
                candidate.dry_run_success = False
                candidate.dry_run_error = result.stderr[:500]

                # Check for OOM
                if "memory" in result.stderr.lower() or "oom" in result.stderr.lower():
                    print(f"    Failed: OOM")
                else:
                    print(f"    Failed: {result.stderr[:100]}")

        except subprocess.TimeoutExpired:
            candidate.dry_run_success = False
            candidate.dry_run_error = "Timeout"
            print(f"    Failed: Timeout")

        except Exception as e:
            candidate.dry_run_success = False
            candidate.dry_run_error = str(e)
            print(f"    Failed: {e}")

        finally:
            # Cleanup
            import shutil
            dryrun_path = f"/tmp/gswa_dryrun_{self.seed}"
            if os.path.exists(dryrun_path):
                shutil.rmtree(dryrun_path, ignore_errors=True)

        return candidate

    def score_candidates(self, candidates: List[PlanCandidate]) -> List[PlanCandidate]:
        """Score and rank candidates.

        Scoring formula:
        score = throughput_score * stability_score * effective_batch_factor

        Where:
        - throughput_score = tokens/sec normalized to best
        - stability_score = 1 - (peak_mem / available_mem), clamped to [0, 1]
        - effective_batch_factor = sqrt(effective_batch_size) normalized
        """
        successful = [c for c in candidates if c.dry_run_success]

        if not successful:
            return candidates

        # Normalize throughput
        max_throughput = max(c.estimated_tokens_per_sec for c in successful)
        max_effective = max(c.effective_batch_size for c in successful)

        for c in successful:
            # Throughput score (0-1)
            c.throughput_score = (
                c.estimated_tokens_per_sec / max_throughput
                if max_throughput > 0 else 0
            )

            # Stability score (1 = lots of margin, 0 = at limit)
            mem_ratio = c.estimated_peak_memory_gb / self.available_memory_gb
            c.stability_score = max(0, min(1, 1 - mem_ratio / self.memory_margin))

            # Effective batch factor (larger effective batch = better gradients)
            batch_factor = (c.effective_batch_size / max_effective) ** 0.5

            # Final score
            c.final_score = (
                c.throughput_score * 0.3 +
                c.stability_score * 0.4 +
                batch_factor * 0.3
            )

        # Sort by score
        successful.sort(key=lambda c: c.final_score, reverse=True)

        return successful

    def run(
        self,
        max_candidates: int = 5,
        skip_dry_run: bool = False,
    ) -> PlannerResult:
        """Run the planning process.

        Args:
            max_candidates: Maximum candidates to test
            skip_dry_run: Skip actual dry-runs (use estimates only)

        Returns:
            PlannerResult with best plan and all candidates
        """
        self.result.planner_started_at = datetime.now().isoformat()
        start_time = time.time()

        print("\n" + "=" * 60)
        print("Training Plan Selection")
        print("=" * 60)
        print(f"\nAvailable memory: {self.available_memory_gb:.1f} GB")
        print(f"Safety margin: {self.memory_margin * 100:.0f}%")
        print(f"Dry-run steps: {self.dry_run_steps}")

        # Generate candidates
        print("\nGenerating candidate configurations...")
        candidates = self.generate_candidates()
        print(f"  Generated {len(candidates)} candidates")

        # Limit candidates for testing
        candidates = candidates[:max_candidates]
        print(f"  Testing top {len(candidates)} candidates")

        if not skip_dry_run:
            print("\nRunning dry-runs...")
            for i, candidate in enumerate(candidates):
                print(f"\n[{i+1}/{len(candidates)}]", end="")
                self.run_dry_run(candidate)
        else:
            print("\nSkipping dry-runs (using estimates only)")
            for c in candidates:
                c.dry_run_success = True
                c.estimated_tokens_per_sec = 100 * c.batch_size / c.max_seq_length * 1024
                c.estimated_peak_memory_gb = self._estimate_memory(
                    c.batch_size, c.max_seq_length, c.lora_rank
                )

        # Score and select best
        print("\nScoring candidates...")
        scored = self.score_candidates(candidates)

        self.result.candidates = scored

        if scored and scored[0].dry_run_success:
            self.result.best_plan = scored[0]
            print(f"\nBest plan selected:")
            print(f"  {self.result.best_plan}")
            print(f"  Score: {self.result.best_plan.final_score:.3f}")
            print(f"  Throughput: {self.result.best_plan.estimated_tokens_per_sec:.1f} tok/s")
            print(f"  Memory: {self.result.best_plan.estimated_peak_memory_gb:.1f} GB")
        else:
            # Fallback to conservative plan
            self.result.best_plan = PlanCandidate(
                batch_size=1,
                max_seq_length=512,
                num_layers=4,
                grad_accum_steps=8,
                eval_batch_size=1,
                lora_rank=8,
                effective_batch_size=8,
            )
            print("\nNo successful candidates, using conservative fallback")

        self.result.planner_ended_at = datetime.now().isoformat()
        self.result.total_dry_run_time_sec = time.time() - start_time

        return self.result

    def print_results_table(self):
        """Print a formatted table of all candidates."""
        print("\n" + "-" * 80)
        print("Candidate Comparison")
        print("-" * 80)
        print(f"{'#':<3} {'batch':<6} {'seq_len':<8} {'layers':<7} {'accum':<6} "
              f"{'tok/s':<8} {'mem_gb':<8} {'score':<7} {'status':<8}")
        print("-" * 80)

        for i, c in enumerate(self.result.candidates, 1):
            status = "OK" if c.dry_run_success else "FAIL"
            if c == self.result.best_plan:
                status = "BEST"

            print(f"{i:<3} {c.batch_size:<6} {c.max_seq_length:<8} {c.num_layers:<7} "
                  f"{c.grad_accum_steps:<6} {c.estimated_tokens_per_sec:<8.1f} "
                  f"{c.estimated_peak_memory_gb:<8.1f} {c.final_score:<7.3f} {status:<8}")


def select_best_plan(
    model_id: str,
    training_data: str,
    available_memory_gb: float,
    memory_margin: float = 0.8,
    dry_run_steps: int = 10,
    max_candidates: int = 5,
    skip_dry_run: bool = False,
    seed: int = 42,
) -> Tuple[PlanCandidate, PlannerResult]:
    """Convenience function to select the best training plan.

    Returns:
        Tuple of (best_plan, full_result)
    """
    planner = TrainingPlanner(
        model_id=model_id,
        training_data=training_data,
        available_memory_gb=available_memory_gb,
        memory_margin=memory_margin,
        dry_run_steps=dry_run_steps,
        seed=seed,
    )

    result = planner.run(max_candidates=max_candidates, skip_dry_run=skip_dry_run)
    planner.print_results_table()

    return result.best_plan, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Plan Selector")
    parser.add_argument("--model", default="mlx-community/Mistral-7B-Instruct-v0.2-4bit")
    parser.add_argument("--data", default="./data/training")
    parser.add_argument("--memory-gb", type=float, default=16.0)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--dry-run-steps", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    best, result = select_best_plan(
        model_id=args.model,
        training_data=args.data,
        available_memory_gb=args.memory_gb,
        memory_margin=args.margin,
        dry_run_steps=args.dry_run_steps,
        max_candidates=args.max_candidates,
        skip_dry_run=args.skip_dry_run,
        seed=args.seed,
    )

    if args.output:
        result.save(args.output)
        print(f"\nResults saved to: {args.output}")
