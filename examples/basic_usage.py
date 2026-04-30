from __future__ import annotations

import torch

from optimizers import BlockwiseConsensusDirectionOptimizer


def main() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    )
    optimizer = BlockwiseConsensusDirectionOptimizer(model.parameters(), lr=1.5e-2)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(16, 32)
    y = torch.randint(0, 10, (16,))

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()

    print("loss:", float(loss.item()))
    print("diagnostics:", optimizer.latest_diagnostics())


if __name__ == "__main__":
    main()
