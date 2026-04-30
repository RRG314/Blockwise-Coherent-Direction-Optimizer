from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer  # noqa: E402


@dataclass
class RunSummary:
    optimizer: str
    best_val_loss: float
    best_val_accuracy: float


def _make_model(input_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )


def _prepare_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = load_breast_cancer()
    x_train, x_val, y_train, y_val = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=31,
        stratify=dataset.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )


def _train(optimizer_name: str) -> RunSummary:
    torch.manual_seed(31)
    np.random.seed(31)
    x_train, y_train, x_val, y_val = _prepare_data()
    model = _make_model(x_train.shape[1])
    if optimizer_name == "bcdo":
        optimizer = BlockwiseConsensusDirectionOptimizer(model.parameters(), lr=0.015)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    else:
        raise ValueError(optimizer_name)
    criterion = torch.nn.BCEWithLogitsLoss()
    batch_size = 32
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for _epoch in range(20):
        indices = torch.randperm(x_train.shape[0])
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if isinstance(optimizer, BlockwiseConsensusDirectionOptimizer):
                optimizer.set_current_loss(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val).item()
            preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = float((preds.eq(y_val)).float().mean().item())
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

    return RunSummary(
        optimizer="BCDO" if optimizer_name == "bcdo" else "AdamW",
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_acc,
    )


def main() -> None:
    summaries = [_train("bcdo"), _train("adamw")]
    print("BCDO versus AdamW comparison")
    for summary in summaries:
        print(
            f"{summary.optimizer}: "
            f"best_val_loss={summary.best_val_loss:.6f}, "
            f"best_val_accuracy={summary.best_val_accuracy:.6f}"
        )


if __name__ == "__main__":
    main()
