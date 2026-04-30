from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research.block_direction_v4_fast_suite import (  # noqa: E402
    block_direction_v4_fast_default_config,
    export_block_direction_v4_fast_report,
    run_block_direction_v4_fast_ablation,
    run_block_direction_v4_fast_benchmarks,
    run_block_direction_v4_fast_smoke,
    run_block_direction_v4_fast_tuning,
    write_block_direction_v4_fast_current_state,
    write_block_direction_v4_fast_literature_scan,
    write_block_direction_v4_fast_math_definition,
)
from optimizers import BlockDirectionOptimizerV4Fast  # noqa: E402
from optimizers.optimizer_utils import set_global_seed  # noqa: E402


class TinyRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 12),
            torch.nn.Tanh(),
            torch.nn.Linear(12, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(20, 4)
    y = 0.5 * x[:, :1] - 0.25 * x[:, 1:2] + 0.1 * x[:, 2:3]
    return x, y


def _make_conv_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 3, (16,))
    return x, y


def _minimal_suite_config(tmp_path: Path) -> dict[str, object]:
    config = block_direction_v4_fast_default_config()
    config.update(
        {
            "output_dir": str(tmp_path),
            "device": "cpu",
            "seeds": [11],
            "search_budget": 1,
            "smoke_seeds": [11],
            "smoke_epoch_scale": 0.10,
            "tuning_epoch_scale": 0.10,
            "benchmark_epoch_scale": 0.12,
            "ablation_epoch_scale": 0.10,
            "optimizers": [
                "block_direction_optimizer_v4_fast",
                "block_direction_optimizer_v4_fast_legacy",
                "block_direction_optimizer_v42",
                "adamw",
                "rmsprop",
            ],
            "smoke_tasks": ["oscillatory_valley"],
            "smoke_optimizers": ["block_direction_optimizer_v4_fast", "adamw"],
            "tuning_tasks": ["oscillatory_valley", "breast_cancer_mlp", "digits_cnn"],
            "benchmark_tasks": ["breast_cancer_mlp", "digits_cnn", "oscillatory_valley"],
            "ablation_tasks": ["digits_cnn", "oscillatory_valley"],
            "use_tuning_results": True,
        }
    )
    return config


def test_block_direction_v4_fast_initializes_and_steps() -> None:
    set_global_seed(123)
    model = TinyRegressor()
    optimizer = BlockDirectionOptimizerV4Fast(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    before = [param.detach().clone() for param in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()
    after = [param.detach().clone() for param in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    diagnostics = optimizer.latest_diagnostics()
    assert diagnostics["selected_candidate_type"] in {
        "gradient",
        "stable_consensus",
        "trusted_direction",
        "low_rank_matrix",
    }
    assert {
        "selected_candidate_type",
        "trust_score",
        "recovery_score",
        "coherence_score",
        "conflict_score",
        "oscillation_score",
        "consensus_strength",
        "block_energy",
        "block_step_norm",
        "fallback_rate",
        "filter_support",
        "conv_step_multiplier",
    }.issubset(diagnostics.keys())


def test_block_direction_v4_fast_state_dict_round_trip() -> None:
    set_global_seed(124)
    model = TinyRegressor()
    optimizer = BlockDirectionOptimizerV4Fast(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()

    state = optimizer.state_dict()

    model_2 = TinyRegressor()
    optimizer_2 = BlockDirectionOptimizerV4Fast(model_2.parameters(), lr=2e-3)
    optimizer_2.load_state_dict(state)
    assert optimizer_2.state_dict()["state"]
    assert optimizer_2.latest_diagnostics() == {"optimizer": "BlockDirectionOptimizerV4Fast"}


def test_block_direction_v4_fast_no_nans_after_multiple_steps() -> None:
    set_global_seed(125)
    model = TinyRegressor()
    optimizer = BlockDirectionOptimizerV4Fast(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        optimizer.set_current_loss(loss.item())
        loss.backward()
        optimizer.step()
    for param in model.parameters():
        assert torch.isfinite(param).all()
    diagnostics = optimizer.latest_diagnostics()
    assert torch.isfinite(torch.tensor(float(diagnostics["block_energy"])))


def test_block_direction_v4_fast_runs_on_conv_filters() -> None:
    set_global_seed(456)
    model = TinyConvClassifier()
    optimizer = BlockDirectionOptimizerV4Fast(model.parameters(), lr=2e-3)
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
    assert "filter_support" in diagnostics


def test_block_direction_v4_fast_runs_on_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        return
    set_global_seed(789)
    device = torch.device("cuda")
    model = TinyRegressor().to(device)
    optimizer = BlockDirectionOptimizerV4Fast(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(float(loss.detach().cpu().item()))
    loss.backward()
    optimizer.step()
    diagnostics = optimizer.latest_diagnostics()
    assert diagnostics["selected_candidate_type"] in {
        "gradient",
        "stable_consensus",
        "trusted_direction",
        "low_rank_matrix",
    }


def test_block_direction_v4_fast_suite_outputs(tmp_path: Path) -> None:
    config = _minimal_suite_config(tmp_path)
    write_block_direction_v4_fast_current_state(tmp_path)
    write_block_direction_v4_fast_literature_scan(tmp_path)
    write_block_direction_v4_fast_math_definition(tmp_path)
    smoke = run_block_direction_v4_fast_smoke(config)
    tuning = run_block_direction_v4_fast_tuning(config)
    bench = run_block_direction_v4_fast_benchmarks(config)
    ablation = run_block_direction_v4_fast_ablation(config)
    export_block_direction_v4_fast_report(tmp_path)

    assert not smoke.empty
    assert not tuning.empty
    assert not bench.empty
    assert not ablation.empty
    assert (tmp_path / "current_state.md").exists()
    assert (tmp_path / "literature_scan.md").exists()
    assert (tmp_path / "literature_matrix.csv").exists()
    assert (tmp_path / "math_definition.md").exists()
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "tuning_results.csv").exists()
    assert (tmp_path / "ablation_results.csv").exists()
    assert (tmp_path / "best_by_task.csv").exists()
    assert (tmp_path / "win_flags.csv").exists()
    assert (tmp_path / "final_report.md").exists()

    benchmark = pd.read_csv(tmp_path / "benchmark_results.csv")
    assert {
        "optimizer",
        "task",
        "suite",
        "final_val_loss",
        "best_val_loss",
        "selection_score",
        "runtime_per_step_ms",
        "optimizer_state_mb",
    }.issubset(benchmark.columns)


def test_block_direction_v4_fast_mps_step_if_available() -> None:
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    set_global_seed(321)
    device = torch.device("mps")
    model = TinyRegressor().to(device)
    optimizer = BlockDirectionOptimizerV4Fast(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(float(loss.detach().cpu().item()))
    loss.backward()
    optimizer.step()
    diagnostics = optimizer.latest_diagnostics()
    assert "block_energy" in diagnostics
