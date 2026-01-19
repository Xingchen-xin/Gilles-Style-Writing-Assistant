"""
Unit tests for GSWA training modules.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gswa.training.preprocessor import DataPreprocessor, PreprocessStats
from gswa.training.run_manager import RunManager, RunConfig
from gswa.training.planner import PlanCandidate, TrainingPlanner
from gswa.training.logger import TrainingLogger
from gswa.training.hardware import HardwareDetector


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_alpaca_data(self, temp_dir):
        """Create sample Alpaca-format data."""
        data = [
            {
                "instruction": "Rewrite this text",
                "input": "This is a short input.",
                "output": "This is a short output.",
            },
            {
                "instruction": "Improve clarity",
                "input": "Sample input text for testing.",
                "output": " ".join(["word"] * 1000),  # Long output
            },
            {
                "instruction": "Polish the text",
                "input": "",
                "output": "A medium length response with several sentences. " * 10,
            },
        ]

        input_file = Path(temp_dir) / "input.jsonl"
        with open(input_file, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')

        return str(input_file)

    def test_estimate_tokens(self):
        """Test token estimation."""
        preprocessor = DataPreprocessor()

        # Empty string
        assert preprocessor.estimate_tokens("") == 0

        # Short text
        tokens = preprocessor.estimate_tokens("Hello world")
        assert tokens > 0 and tokens < 10

        # Longer text
        long_text = " ".join(["word"] * 100)
        tokens = preprocessor.estimate_tokens(long_text)
        assert 80 < tokens < 150  # Roughly 1.3 tokens per word

    def test_analyze_data(self, sample_alpaca_data):
        """Test data analysis."""
        preprocessor = DataPreprocessor(max_tokens=512)
        stats = preprocessor.analyze_data(sample_alpaca_data)

        assert stats.total_entries_before == 3
        assert stats.format_detected == "alpaca"
        assert stats.min_tokens_before > 0
        assert stats.max_tokens_before > stats.min_tokens_before

    def test_preprocess_splits_long_sequences(self, sample_alpaca_data, temp_dir):
        """Test that preprocessing splits long sequences."""
        output_file = Path(temp_dir) / "output.jsonl"

        preprocessor = DataPreprocessor(max_tokens=256, overlap_tokens=20)
        stats = preprocessor.preprocess(sample_alpaca_data, str(output_file))

        # Should have more entries after splitting
        assert stats.total_entries_after >= stats.total_entries_before
        assert stats.max_tokens_after <= 256 * 1.5  # Allow some tolerance
        assert stats.truncation_pct_after < stats.truncation_pct_before or stats.truncation_pct_before == 0

    def test_split_by_sentence(self):
        """Test sentence-based splitting."""
        preprocessor = DataPreprocessor(max_tokens=50)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = preprocessor._split_by_sentence(text, 50)

        assert len(chunks) >= 1
        for chunk in chunks:
            # Each chunk should have complete sentences
            assert chunk.endswith('.') or chunk.endswith('. ')

    def test_split_by_paragraph(self):
        """Test paragraph-based splitting."""
        preprocessor = DataPreprocessor(max_tokens=100)

        text = "First paragraph with content.\n\nSecond paragraph here.\n\nThird paragraph."
        chunks = preprocessor._split_by_paragraph(text, 100)

        assert len(chunks) >= 1

    def test_generate_report(self, sample_alpaca_data, temp_dir):
        """Test report generation."""
        output_file = Path(temp_dir) / "output.jsonl"

        preprocessor = DataPreprocessor(max_tokens=512)
        stats = preprocessor.preprocess(sample_alpaca_data, str(output_file))

        report = preprocessor.generate_report(stats)

        assert "Data Preprocessing Report" in report
        assert "Before Preprocessing" in report
        assert "After Preprocessing" in report


class TestRunManager:
    """Tests for RunManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_create_run(self, temp_dir):
        """Test run creation."""
        manager = RunManager(temp_dir)
        config = RunConfig(
            model_id="test-model",
            batch_size=2,
            max_seq_length=1024,
        )

        run_dir = manager.create_run(config, run_name="test-run")

        assert run_dir.exists()
        assert (run_dir / "config").exists()
        assert (run_dir / "logs").exists()
        assert (run_dir / "plots").exists()
        assert (run_dir / "config" / "run_config.json").exists()

    def test_oom_fallback(self, temp_dir):
        """Test OOM fallback logic."""
        manager = RunManager(temp_dir)
        config = RunConfig(
            batch_size=4,
            eval_batch_size=2,
            max_seq_length=2048,
            num_layers=16,
            enable_oom_fallback=True,
        )

        manager.create_run(config)

        # First fallback: reduce eval_batch_size
        new_config = manager.apply_oom_fallback("OOM error")
        assert new_config.eval_batch_size == 1
        assert new_config.batch_size == 4

        # Second fallback: reduce batch_size (4 -> 2)
        new_config = manager.apply_oom_fallback("OOM error")
        assert new_config.batch_size == 2

        # Third fallback: reduce batch_size again (2 -> 1)
        new_config = manager.apply_oom_fallback("OOM error")
        assert new_config.batch_size == 1

        # Fourth fallback: reduce max_seq_length
        new_config = manager.apply_oom_fallback("OOM error")
        assert new_config.max_seq_length < 2048

    def test_config_serialization(self, temp_dir):
        """Test config save/load."""
        config = RunConfig(
            model_id="test-model",
            batch_size=2,
            learning_rate=1e-5,
        )

        config_path = Path(temp_dir) / "config.json"
        config.save(str(config_path))

        loaded = RunConfig.from_file(str(config_path))
        assert loaded.model_id == config.model_id
        assert loaded.batch_size == config.batch_size
        assert loaded.learning_rate == config.learning_rate


class TestPlanCandidate:
    """Tests for PlanCandidate and scoring."""

    def test_candidate_creation(self):
        """Test candidate creation."""
        candidate = PlanCandidate(
            batch_size=2,
            max_seq_length=1024,
            num_layers=8,
            grad_accum_steps=4,
        )

        assert candidate.batch_size == 2
        assert candidate.effective_batch_size == 0  # Not yet calculated

    def test_candidate_serialization(self):
        """Test candidate to_dict."""
        candidate = PlanCandidate(
            batch_size=2,
            max_seq_length=1024,
            dry_run_success=True,
        )

        data = candidate.to_dict()
        assert data["batch_size"] == 2
        assert data["max_seq_length"] == 1024
        assert data["dry_run_success"] is True


class TestTrainingPlanner:
    """Tests for TrainingPlanner."""

    def test_generate_candidates(self):
        """Test candidate generation."""
        planner = TrainingPlanner(
            model_id="test-model",
            training_data="./data",
            available_memory_gb=16.0,
        )

        candidates = planner.generate_candidates(
            seq_lengths=[512, 1024],
            batch_sizes=[1, 2],
            grad_accum=[1, 2],
            lora_ranks=[8],
        )

        assert len(candidates) > 0
        # All candidates should have valid values
        for c in candidates:
            assert c.batch_size >= 1
            assert c.max_seq_length >= 512
            assert c.num_layers >= 4

    def test_memory_estimation(self):
        """Test memory estimation."""
        planner = TrainingPlanner(
            model_id="test-model",
            training_data="./data",
            available_memory_gb=16.0,
        )

        # Larger batch/seq should use more memory
        mem_small = planner._estimate_memory(1, 512, 8)
        mem_large = planner._estimate_memory(4, 2048, 16)

        assert mem_large > mem_small

    def test_score_candidates(self):
        """Test candidate scoring."""
        planner = TrainingPlanner(
            model_id="test-model",
            training_data="./data",
            available_memory_gb=16.0,
        )

        candidates = [
            PlanCandidate(
                batch_size=1,
                max_seq_length=1024,
                dry_run_success=True,
                estimated_tokens_per_sec=100,
                estimated_peak_memory_gb=8,
                effective_batch_size=4,
            ),
            PlanCandidate(
                batch_size=2,
                max_seq_length=1024,
                dry_run_success=True,
                estimated_tokens_per_sec=150,
                estimated_peak_memory_gb=12,
                effective_batch_size=8,
            ),
            PlanCandidate(
                batch_size=1,
                max_seq_length=512,
                dry_run_success=False,  # Failed
                estimated_tokens_per_sec=50,
                estimated_peak_memory_gb=4,
                effective_batch_size=2,
            ),
        ]

        scored = planner.score_candidates(candidates)

        # Best should be first (excluding failed)
        assert scored[0].dry_run_success is True
        assert scored[0].final_score > 0


class TestTrainingLogger:
    """Tests for TrainingLogger."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_log_step(self, temp_dir):
        """Test logging training steps."""
        with TrainingLogger(temp_dir, "test-run") as logger:
            for i in range(10):
                logger.log_step(
                    step=i,
                    train_loss=2.0 - i * 0.1,
                    learning_rate=1e-5,
                    tokens_per_sec=100 + i,
                    trained_tokens=i * 1000,
                )

            assert len(logger.train_steps) == 10
            assert logger.train_steps[-1].train_loss < logger.train_steps[0].train_loss

    def test_log_eval(self, temp_dir):
        """Test logging eval steps."""
        with TrainingLogger(temp_dir, "test-run") as logger:
            logger.log_eval(
                step=100,
                eval_loss=1.5,
                eval_duration_sec=5.0,
                eval_samples=100,
            )

            assert len(logger.eval_steps) == 1
            assert logger.best_eval_loss == 1.5

    def test_log_oom_event(self, temp_dir):
        """Test logging OOM events."""
        with TrainingLogger(temp_dir, "test-run") as logger:
            logger.log_oom_event(
                step=50,
                error_message="Metal OOM",
                config_before={"batch_size": 4},
                config_after={"batch_size": 2},
                fallback_action="Reduced batch size",
            )

            assert len(logger.events) == 1
            assert logger.events[0]["type"] == "oom_fallback"

    def test_export_for_plotting(self, temp_dir):
        """Test data export."""
        with TrainingLogger(temp_dir, "test-run") as logger:
            for i in range(5):
                logger.log_step(
                    step=i,
                    train_loss=2.0 - i * 0.1,
                    learning_rate=1e-5,
                    tokens_per_sec=100,
                    trained_tokens=i * 1000,
                )

            data = logger.export_for_plotting()

            assert "steps" in data
            assert "train_loss" in data
            assert len(data["steps"]) == 5

    def test_load_from_logs(self, temp_dir):
        """Test loading from log files."""
        # Create some logs
        with TrainingLogger(temp_dir, "test-run") as logger:
            for i in range(5):
                logger.log_step(
                    step=i,
                    train_loss=2.0 - i * 0.1,
                    learning_rate=1e-5,
                    tokens_per_sec=100,
                    trained_tokens=i * 1000,
                )

        # Load them back
        loaded = TrainingLogger.load_from_logs(temp_dir)
        assert len(loaded.train_steps) == 5


class TestHardwareDetector:
    """Tests for HardwareDetector."""

    def test_detect_basic(self):
        """Test basic detection."""
        detector = HardwareDetector()
        info = detector.detect()

        # Should always have OS info
        assert info.os_name != ""
        assert info.cpu_cores > 0
        assert info.total_memory_gb > 0

    def test_recommended_settings(self):
        """Test that recommendations are sane."""
        detector = HardwareDetector()
        info = detector.detect()

        # All recommendations should be positive
        assert info.recommended_batch_size >= 1
        assert info.recommended_max_seq_length >= 256
        assert info.recommended_num_layers >= 4
        assert info.recommended_eval_batch_size >= 1

    def test_safe_config(self):
        """Test get_safe_config."""
        detector = HardwareDetector()
        detector.detect()

        config = detector.get_safe_config(margin=0.5)

        assert config["batch_size"] >= 1
        assert config["max_seq_length"] >= 256
        assert config["num_layers"] >= 4

    def test_hardware_info_serialization(self):
        """Test HardwareInfo serialization."""
        detector = HardwareDetector()
        info = detector.detect()

        # To dict and back
        data = info.to_dict()
        assert "os_name" in data
        assert "total_memory_gb" in data

        # To JSON
        json_str = info.to_json()
        parsed = json.loads(json_str)
        assert parsed["os_name"] == info.os_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
