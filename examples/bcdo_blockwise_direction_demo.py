from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer  # noqa: E402


REPORT_DIR = ROOT / "reports" / "bcdo_mainline"


def _make_low_rank_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    input_dim = 64
    output_dim = 64
    rank = 4
    left = torch.randn(output_dim, rank)
    right = torch.randn(rank, input_dim)
    target_weight = 0.1 * (left @ right) / rank

    x_train = torch.randn(384, input_dim)
    y_train = x_train @ target_weight.t()
    x_val = torch.randn(128, input_dim)
    y_val = x_val @ target_weight.t()
    return x_train, y_train, x_val, y_val


def _run_bcdo() -> dict[str, list[float] | float]:
    x_train, y_train, x_val, y_val = _make_low_rank_problem()
    model = torch.nn.Linear(x_train.shape[1], y_train.shape[1], bias=False)
    optimizer = BlockwiseConsensusDirectionOptimizer(
        model.parameters(),
        lr=0.01,
        small_matrix_cutoff=0,
    )
    criterion = torch.nn.MSELoss()
    losses: list[float] = []
    best_val_loss = float("inf")

    for _epoch in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        optimizer.set_current_loss(loss.item())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        losses.append(val_loss)
        best_val_loss = min(best_val_loss, val_loss)

    diagnostics = optimizer.latest_diagnostics()
    return {
        "losses": losses,
        "best_val_loss": best_val_loss,
        "final_val_loss": losses[-1],
        "trust_score": float(diagnostics.get("trust_score", float("nan"))),
        "consensus_strength": float(diagnostics.get("consensus_strength", float("nan"))),
        "selected_candidate_type": diagnostics.get("selected_candidate_type", "unknown"),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bcdo_run = _run_bcdo()

    plt.figure(figsize=(9, 4.5))
    plt.plot(bcdo_run["losses"], label="BCDO")
    plt.title("Low-rank matrix demo: BCDO validation loss")
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.legend()
    plt.tight_layout()
    output_path = REPORT_DIR / "example_blockwise_direction_demo.png"
    plt.savefig(output_path, dpi=180)
    plt.close()

    print("BCDO blockwise direction demo")
    print(
        f"best_val_loss={bcdo_run['best_val_loss']:.6f}, "
        f"final_val_loss={bcdo_run['final_val_loss']:.6f}, "
        f"selected_candidate_type={bcdo_run['selected_candidate_type']}, "
        f"trust_score={bcdo_run['trust_score']}, "
        f"consensus_strength={bcdo_run['consensus_strength']}"
    )
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
