from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research.block_direction_v42_suite import (  # noqa: E402
    block_direction_v42_default_config,
    export_block_direction_v42_report,
    run_block_direction_v42_ablation,
    run_block_direction_v42_benchmarks,
    run_block_direction_v42_smoke,
    run_block_direction_v42_tuning,
    write_block_direction_v42_current_state,
    write_block_direction_v42_literature_scan,
    write_block_direction_v42_math_definition,
)
from optimizers import BlockDirectionOptimizerV42  # noqa: E402
from optimizers.optimizer_utils import set_global_seed  # noqa: E402


class TinyConvClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 6, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(6 * 8 * 8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_conv_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 3, (16,))
    return x, y


def _minimal_suite_config(tmp_path: Path) -> dict[str, object]:
    config = block_direction_v42_default_config()
    config.update(
        {
            "output_dir": str(tmp_path),
            "device": "cpu",
            "seeds": [11],
            "search_budget": 1,
            "smoke_seeds": [11],
            "smoke_epoch_scale": 0.08,
            "tuning_epoch_scale": 0.10,
            "benchmark_epoch_scale": 0.12,
            "ablation_epoch_scale": 0.10,
            "optimizers": [
                "block_direction_optimizer_v42",
                "block_direction_optimizer_v4_fast",
                "adamw",
                "rmsprop",
            ],
            "smoke_tasks": ["digits_cnn"],
            "smoke_optimizers": ["block_direction_optimizer_v42", "adamw"],
            "tuning_tasks": ["digits_cnn", "breast_cancer_mlp"],
            "benchmark_tasks": ["digits_cnn", "breast_cancer_mlp"],
            "ablation_tasks": ["digits_cnn"],
            "use_tuning_results": True,
        }
    )
    return config


def test_block_direction_v42_runs_and_logs_conv_diagnostics() -> None:
    set_global_seed(123)
    model = TinyConvClassifier()
    optimizer = BlockDirectionOptimizerV42(model.parameters(), lr=2e-3)
    criterion = torch.nn.CrossEntropyLoss()
    x, y = _make_conv_batch()
    before = [param.detach().clone() for param in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()
    after = [param.detach().clone() for param in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    diagnostics = optimizer.latest_diagnostics()
    assert {"filter_support", "conv_trust_bonus", "conv_step_multiplier", "selected_candidate_type"}.issubset(diagnostics.keys())


def test_block_direction_v42_suite_outputs(tmp_path: Path) -> None:
    config = _minimal_suite_config(tmp_path)
    write_block_direction_v42_current_state(tmp_path)
    write_block_direction_v42_literature_scan(tmp_path)
    write_block_direction_v42_math_definition(tmp_path)
    smoke = run_block_direction_v42_smoke(config)
    tuning = run_block_direction_v42_tuning(config)
    bench = run_block_direction_v42_benchmarks(config)
    ablation = run_block_direction_v42_ablation(config)
    export_block_direction_v42_report(tmp_path)

    assert not smoke.empty
    assert not tuning.empty
    assert not bench.empty
    assert not ablation.empty
    benchmark = pd.read_csv(tmp_path / "benchmark_results.csv")
    assert {"optimizer", "task", "best_val_loss", "runtime_per_step_ms", "mean_filter_support", "mean_conv_trust_bonus", "mean_conv_step_multiplier"}.issubset(benchmark.columns)
    assert (tmp_path / "final_report.md").exists()
